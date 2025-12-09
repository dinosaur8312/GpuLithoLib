#include "RawContourDetectEngine.cuh"
#include "GpuKernelProfiler.cuh"
#include "LayerImpl.h"
#include "CommonRenderUtils.cuh"
#include "VisualizationUtils.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/distance.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// Platform-specific CUB include
#ifdef __HIP_PLATFORM_AMD__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/device/device_run_length_encode.cuh>
#endif

namespace GpuLithoLib {

// External profiler instance (owned by GpuLithoEngine)
extern GpuKernelProfiler* g_kernelProfiler;

// ============================================================================
// RawContourDetectEngine Implementation
// ============================================================================

RawContourDetectEngine::RawContourDetectEngine() {}

RawContourDetectEngine::~RawContourDetectEngine() {}

// ============================================================================
// Contour Detection Methods
// ============================================================================

std::vector<std::vector<cv::Point>> RawContourDetectEngine::detectRawContours(
    LayerImpl* outputLayer,
    OperationType opType,
    unsigned int currentGridWidth,
    unsigned int currentGridHeight) {
    std::vector<std::vector<cv::Point>> contours;

    if (!outputLayer) {
        return contours;
    }

    // NOTE: The GPU kernel approach for extracting contour pixels
    // This creates a bitmap with only contour pixels rendered

    // Ensure we have a contour bitmap
    auto contourLayer = std::make_unique<LayerImpl>();
    contourLayer->ensureBitmapAllocated(currentGridWidth, currentGridHeight);
    contourLayer->clearBitmap();

    // Extract contours using GPU kernel
    const int chunkDim = 30;
    dim3 blockSize(32, 32);
    dim3 gridSize(iDivUp(currentGridWidth, chunkDim), iDivUp(currentGridHeight, chunkDim));

    // Time the extractContours kernel
    gpuEvent_t extractStart, extractStop;
    gpuEventCreate(&extractStart);
    gpuEventCreate(&extractStop);
    gpuEventRecord(extractStart);

    extractContours_kernel<<<gridSize, blockSize>>>(
        outputLayer->d_bitmap,
        contourLayer->d_bitmap,
        currentGridWidth,
        currentGridHeight,
        chunkDim,
        opType);

    CHECK_GPU_ERROR(gpuGetLastError());
    gpuEventRecord(extractStop);
    gpuEventSynchronize(extractStop);

    float extractMs = 0.0f;
    gpuEventElapsedTime(&extractMs, extractStart, extractStop);
    if (g_kernelProfiler) g_kernelProfiler->addExtractContoursTime(extractMs);

    gpuEventDestroy(extractStart);
    gpuEventDestroy(extractStop);

    // Copy to host
    contourLayer->copyBitmapToHost();

    // ========================================================================
    // GPU-based contour pixel sorting for parallel contour tracing
    // ========================================================================
    SortedContourPixels sortedPixels = sortContourPixelsByValue(
        contourLayer->d_bitmap,
        currentGridWidth,
        currentGridHeight);

    // Print statistics about sorted groups
    if (sortedPixels.count > 0) {
        std::cout << "Contour pixel sorting complete:" << std::endl;
        std::cout << "  Total non-zero pixels: " << sortedPixels.count << std::endl;

        // Count unique values (groups) by checking value changes
        thrust::device_vector<unsigned int> uniqueValues(sortedPixels.count);
        auto uniqueEnd = thrust::unique_copy(
            sortedPixels.values.begin(),
            sortedPixels.values.end(),
            uniqueValues.begin());
        unsigned int numGroups = thrust::distance(uniqueValues.begin(), uniqueEnd);
        std::cout << "  Number of unique pixel value groups: " << numGroups << std::endl;
    }

    // ========================================================================
    // GPU Contour Tracing for INTERSECTION operation
    // ========================================================================
    std::vector<std::vector<cv::Point>> gpuContours;
    if (opType == OperationType::INTERSECTION && sortedPixels.count > 0) {
        std::cout << "Using GPU contour tracing for INTERSECTION operation" << std::endl;

        // Trace contours using GPU parallel algorithm
        gpuContours = traceContoursGPU(
            sortedPixels,
            contourLayer->d_bitmap,
            outputLayer->d_bitmap,  // Pass overlay bitmap for polygon ID collection
            currentGridWidth,
            currentGridHeight);

        std::cout << "GPU tracing found " << gpuContours.size() << " contours" << std::endl;

        // Return GPU contours for INTERSECTION operation
        return gpuContours;
    }

    // For non-INTERSECTION operations, return empty (caller should use detectRawContoursCPU)
    return contours;
}

// ============================================================================
// CPU Contour Detection Implementation (OpenCV findContours)
// ============================================================================

std::vector<std::vector<cv::Point>> RawContourDetectEngine::detectRawContoursCPU(
    LayerImpl* outputLayer,
    OperationType opType,
    unsigned int currentGridWidth,
    unsigned int currentGridHeight) {

    std::vector<std::vector<cv::Point>> contours;

    if (!outputLayer) {
        return contours;
    }

    // Ensure output bitmap is on host
    outputLayer->copyBitmapToHost();

    // Create binary image directly from outputLayer bitmap based on operation type
    cv::Mat binaryImage(currentGridHeight, currentGridWidth, CV_8UC1);
    for (unsigned int y = 0; y < currentGridHeight; ++y) {
        for (unsigned int x = 0; x < currentGridWidth; ++x) {
            unsigned int idx = y * currentGridWidth + x;
            unsigned int val = outputLayer->h_bitmap[idx];

            switch(opType) {
                case OperationType::OFFSET:
                {
                    binaryImage.at<uchar>(y, x) = (val & 0xFFFF) ? 255 : 0;
                    break;
                }
                case OperationType::INTERSECTION:
                {
                    binaryImage.at<uchar>(y, x) = ((val & 0xFFFF) && (val & 0xFFFF0000)) ? 255 : 0;
                    break;
                }
                case OperationType::UNION:
                {
                    binaryImage.at<uchar>(y, x) = ((val & 0xFFFF) || (val & 0xFFFF0000)) ? 255 : 0;
                    break;
                }
                case OperationType::DIFFERENCE:
                {
                    binaryImage.at<uchar>(y, x) = ((val & 0xFFFF) && !(val & 0xFFFF0000)) ? 255 : 0;
                    break;
                }
                default:
                {
                    binaryImage.at<uchar>(y, x) = (val > 0) ? 255 : 0;
                    break;
                }
            }
        }
    }

    // Use OpenCV to find contours
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);

    std::cout << "CPU contour detection found " << contours.size() << " contours" << std::endl;

    return contours;
}

// ============================================================================
// GPU Contour Pixel Sorting Implementation
// ============================================================================

SortedContourPixels RawContourDetectEngine::sortContourPixelsByValue(
    const unsigned int* contourBitmap,
    unsigned int width,
    unsigned int height) {

    SortedContourPixels result;

    unsigned int totalPixels = width * height;

    // Step 1: Create index array [0, 1, 2, ..., totalPixels-1]
    thrust::device_vector<unsigned int> d_indices(totalPixels);
    thrust::sequence(d_indices.begin(), d_indices.end(), 0);

    // Step 2: Copy bitmap to device_vector for processing
    thrust::device_vector<unsigned int> d_values(totalPixels);
    thrust::copy(
        thrust::device_pointer_cast(contourBitmap),
        thrust::device_pointer_cast(contourBitmap + totalPixels),
        d_values.begin());

    // Step 3: Remove elements where bitmap value is 0
    auto new_end = thrust::remove_if(
        thrust::make_zip_iterator(thrust::make_tuple(d_values.begin(), d_indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_values.end(), d_indices.end())),
        d_values.begin(),
        thrust::placeholders::_1 == 0);

    // Calculate the number of non-zero pixels
    auto values_new_end = new_end.get_iterator_tuple().template get<0>();
    unsigned int nonZeroCount = thrust::distance(d_values.begin(), values_new_end);

    // Resize vectors to actual size
    d_values.resize(nonZeroCount);
    d_indices.resize(nonZeroCount);

    // Step 4: Sort indices by bitmap values (using values as keys)
    thrust::sort_by_key(d_values.begin(), d_values.end(), d_indices.begin());

    // Step 5: Store results
    result.values = std::move(d_values);
    result.indices = std::move(d_indices);
    result.count = nonZeroCount;

    std::cout << "Sorted " << nonZeroCount << " non-zero contour pixels into groups by value" << std::endl;

    return result;
}

// ============================================================================
// GPU Contour Tracing Implementation
// ============================================================================

std::vector<std::vector<cv::Point>> RawContourDetectEngine::traceContoursGPU(
    const SortedContourPixels& sortedPixels,
    const unsigned int* contourBitmap,
    const unsigned int* overlayBitmap,
    unsigned int width,
    unsigned int height) {

    std::vector<std::vector<cv::Point>> contours;

    if (sortedPixels.count == 0) {
        return contours;
    }

    // Step 1: Identify groups using CUB Run-Length Encode
    thrust::device_vector<unsigned int> d_unique_values(sortedPixels.count);
    thrust::device_vector<unsigned int> d_group_counts(sortedPixels.count);
    thrust::device_vector<unsigned int> d_num_runs(1);

    // Determine temporary storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage, temp_storage_bytes,
        thrust::raw_pointer_cast(sortedPixels.values.data()),
        thrust::raw_pointer_cast(d_unique_values.data()),
        thrust::raw_pointer_cast(d_group_counts.data()),
        thrust::raw_pointer_cast(d_num_runs.data()),
        sortedPixels.count);

    // Allocate temporary storage
    CHECK_GPU_ERROR(gpuMalloc(&d_temp_storage, temp_storage_bytes));

    // Run encoding
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage, temp_storage_bytes,
        thrust::raw_pointer_cast(sortedPixels.values.data()),
        thrust::raw_pointer_cast(d_unique_values.data()),
        thrust::raw_pointer_cast(d_group_counts.data()),
        thrust::raw_pointer_cast(d_num_runs.data()),
        sortedPixels.count);

    // Get number of groups
    unsigned int numGroups;
    CHECK_GPU_ERROR(gpuMemcpy(&numGroups, thrust::raw_pointer_cast(d_num_runs.data()),
                              sizeof(unsigned int), gpuMemcpyDeviceToHost));

    CHECK_GPU_ERROR(gpuFree(d_temp_storage));

    std::cout << "Found " << numGroups << " pixel value groups for tracing" << std::endl;

    if (numGroups == 0) {
        return contours;
    }

    // Resize to actual number of groups
    d_unique_values.resize(numGroups);
    d_group_counts.resize(numGroups);

    // Step 2: Compute start indices using exclusive scan
    thrust::device_vector<unsigned int> d_group_starts(numGroups);
    thrust::exclusive_scan(d_group_counts.begin(), d_group_counts.end(), d_group_starts.begin());

    // Step 3: Allocate visited arrays for border-following algorithm
    // Edge visited array: tracks which edges have been traced
    // Size = (height+1) * (width+1) * 4 (4 directions per vertex)
    size_t edgeVisitedSize = (size_t)(height + 1) * (width + 1) * 4;
    thrust::device_vector<unsigned char> d_edgeVisited(edgeVisitedSize, 0);

    // Pixel processed array: tracks which pixels have been used as starting points
    thrust::device_vector<unsigned char> d_pixelProcessed(width * height, 0);

    // Also keep the old visited array for the original kernel (for fallback/comparison)
    thrust::device_vector<unsigned char> d_visited(sortedPixels.count, 0);

    // Step 4: Allocate output arrays
    const unsigned int maxPointsPerContour = 1<<13; // 8192 points max per contour
    const unsigned int maxIDsPerGroup = 256;
    thrust::device_vector<uint2> d_outputContours(numGroups * maxPointsPerContour);
    thrust::device_vector<unsigned int> d_outputCounts(numGroups, 0);

    // Allocate separate arrays for polygon IDs
    thrust::device_vector<unsigned int> d_subject_ids(numGroups * maxIDsPerGroup);
    thrust::device_vector<unsigned int> d_clipper_ids(numGroups * maxIDsPerGroup);
    thrust::device_vector<unsigned int> d_subject_counts(numGroups, 0);
    thrust::device_vector<unsigned int> d_clipper_counts(numGroups, 0);

    // Step 5: Launch tracing kernel
    // Use border-following algorithm (handles single-pixel protrusions correctly)
    dim3 blockSize(1, 1, 1);
    dim3 gridSize(numGroups, 1, 1);

    gpuEvent_t traceStart, traceStop;
    gpuEventCreate(&traceStart);
    gpuEventCreate(&traceStop);
    gpuEventRecord(traceStart);

    // Use the new border-following kernel
    traceContoursParallelBorder_kernel<<<gridSize, blockSize>>>(
        contourBitmap,
        overlayBitmap,
        thrust::raw_pointer_cast(sortedPixels.indices.data()),
        thrust::raw_pointer_cast(sortedPixels.values.data()),
        thrust::raw_pointer_cast(d_unique_values.data()),
        thrust::raw_pointer_cast(d_group_starts.data()),
        thrust::raw_pointer_cast(d_group_counts.data()),
        numGroups,
        thrust::raw_pointer_cast(d_edgeVisited.data()),
        thrust::raw_pointer_cast(d_pixelProcessed.data()),
        thrust::raw_pointer_cast(d_outputContours.data()),
        thrust::raw_pointer_cast(d_outputCounts.data()),
        thrust::raw_pointer_cast(d_subject_ids.data()),
        thrust::raw_pointer_cast(d_clipper_ids.data()),
        thrust::raw_pointer_cast(d_subject_counts.data()),
        thrust::raw_pointer_cast(d_clipper_counts.data()),
        width,
        height,
        maxPointsPerContour);

    CHECK_GPU_ERROR(gpuGetLastError());
    gpuEventRecord(traceStop);
    gpuEventSynchronize(traceStop);

    float traceMs = 0.0f;
    gpuEventElapsedTime(&traceMs, traceStart, traceStop);
    if (g_kernelProfiler) g_kernelProfiler->addTraceContoursTime(traceMs);

    gpuEventDestroy(traceStart);
    gpuEventDestroy(traceStop);

    // Step 6: Copy results back to host
    thrust::host_vector<uint2> h_outputContours = d_outputContours;
    thrust::host_vector<unsigned int> h_outputCounts = d_outputCounts;
    thrust::host_vector<unsigned int> h_subject_ids = d_subject_ids;
    thrust::host_vector<unsigned int> h_clipper_ids = d_clipper_ids;
    thrust::host_vector<unsigned int> h_subject_counts = d_subject_counts;
    thrust::host_vector<unsigned int> h_clipper_counts = d_clipper_counts;

    // Step 7: Convert to cv::Point format and print polygon IDs
    for (unsigned int g = 0; g < numGroups; ++g) {
        unsigned int totalCount = h_outputCounts[g];
        if (totalCount > 0) {
            unsigned int baseIdx = g * maxPointsPerContour;
            unsigned int i = 0;
            unsigned int contourInGroup = 0;

            while (i < totalCount) {
                unsigned int contourLen = 0;
                unsigned int startIdx = i;

                if (contourInGroup == 0) {
                    while (i < totalCount) {
                        uint2 pt = h_outputContours[baseIdx + i];
                        if (pt.x == 0xFFFFFFFF) {
                            break;
                        }
                        i++;
                    }
                    contourLen = i - startIdx;
                } else {
                    uint2 marker = h_outputContours[baseIdx + i];
                    if (marker.x == 0xFFFFFFFF) {
                        contourLen = marker.y;
                        i++;
                        startIdx = i;
                        i += contourLen;
                    } else {
                        i++;
                        continue;
                    }
                }

                if (contourLen > 0) {
                    std::vector<cv::Point> contour;
                    contour.reserve(contourLen);

                    for (unsigned int j = 0; j < contourLen; ++j) {
                        uint2 pt = h_outputContours[baseIdx + startIdx + j];
                        contour.push_back(cv::Point(pt.x, pt.y));
                    }

                    contours.push_back(std::move(contour));

                    unsigned int subjectCount = h_subject_counts[g];
                    unsigned int clipperCount = h_clipper_counts[g];
                    std::cout << "  Group " << g << " Contour " << contourInGroup
                              << " (" << contourLen << " points): ";
                    std::cout << "Subject IDs [" << subjectCount << "]: ";
                    for (unsigned int k = 0; k < subjectCount; ++k) {
                        std::cout << h_subject_ids[g * maxIDsPerGroup + k];
                        if (k < subjectCount - 1) std::cout << ", ";
                    }
                    std::cout << " | Clipper IDs [" << clipperCount << "]: ";
                    for (unsigned int k = 0; k < clipperCount; ++k) {
                        std::cout << h_clipper_ids[g * maxIDsPerGroup + k];
                        if (k < clipperCount - 1) std::cout << ", ";
                    }
                    std::cout << std::endl;
                }

                contourInGroup++;
            }
        }
    }

    std::cout << "GPU tracing complete: extracted " << contours.size() << " contours" << std::endl;

    // Visualize GPU traced contours using VisualizationUtils
    {
        // Convert thrust::host_vector to std::vector for visualization
        std::vector<uint2> h_contours_vec(h_outputContours.begin(), h_outputContours.end());
        std::vector<unsigned int> h_counts_vec(h_outputCounts.begin(), h_outputCounts.end());
        thrust::host_vector<unsigned int> h_unique_values_thrust(d_unique_values);
        std::vector<unsigned int> h_unique_values_vec(h_unique_values_thrust.begin(), h_unique_values_thrust.end());

        VisualizationUtils::visualizeGPUTracedContours(
            h_contours_vec,
            h_counts_vec,
            h_unique_values_vec,
            numGroups,
            maxPointsPerContour,
            width,
            height,
            cv::Scalar(0, 0, 255),  // Red contours
            "step4_gpu_contour_tracing_visualization.png");
    }

    return contours;
}

// ============================================================================
// GPU Kernels for Contour Detection and Tracing
// ============================================================================

__global__ void extractContours_kernel(
    const unsigned int* inputBitmap,
    unsigned int* contourBitmap,
    const int width,
    const int height,
    const int chunkDim,
    OperationType opType)
{
    __shared__ unsigned int s_isContourPixel[32][32];
    s_isContourPixel[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    uint2 chunkStart = make_uint2(blockIdx.x * chunkDim, blockIdx.y * chunkDim);
    uint2 chunkEnd = make_uint2(chunkStart.x + chunkDim, chunkStart.y + chunkDim);
    uint2 chunkEndClamped = make_uint2(min(chunkEnd.x, width), min(chunkEnd.y, height));

    uint2 apronStart = make_uint2(chunkStart.x - 1, chunkStart.y - 1);
    uint2 apronEnd = make_uint2(chunkEnd.x + 1, chunkEnd.y + 1);

    uint2 apronStartClamped = make_uint2(max(apronStart.x, 0), max(apronStart.y, 0));
    uint2 apronEndClamped = make_uint2(min(apronEnd.x, width), min(apronEnd.y, height));

    int g_ix = apronStart.x + threadIdx.x;
    int g_iy = apronStart.y + threadIdx.y;
    int g_idx = g_iy * width + g_ix;

    if (g_ix >= apronStartClamped.x && g_ix < apronEndClamped.x &&
        g_iy >= apronStartClamped.y && g_iy < apronEndClamped.y) {

        unsigned int pixelValue = inputBitmap[g_idx];
        if (pixelValue > 0) {
            unsigned int subject_id = pixelValue & 0xFFFF;
            unsigned int clipper_id = (pixelValue >> 16) & 0xFFFF;

            switch (opType) {
                case OperationType::INTERSECTION:
                    if (subject_id > 0 && clipper_id > 0) {
                        s_isContourPixel[threadIdx.y][threadIdx.x] = pixelValue;
                    }
                    break;
                case OperationType::UNION:
                    if (subject_id > 0 || clipper_id > 0) {
                        s_isContourPixel[threadIdx.y][threadIdx.x] = pixelValue;
                    }
                    break;
                case OperationType::DIFFERENCE:
                    if (subject_id > 0 && clipper_id == 0) {
                        s_isContourPixel[threadIdx.y][threadIdx.x] = pixelValue;
                    }
                    break;
                case OperationType::XOR:
                    if ((subject_id > 0) != (clipper_id > 0)) {
                        s_isContourPixel[threadIdx.y][threadIdx.x] = pixelValue;
                    }
                    break;
                case OperationType::OFFSET:
                default:
                    s_isContourPixel[threadIdx.y][threadIdx.x] = pixelValue;
                    break;
            }
        }
    }

    __syncthreads();

    if (g_ix < chunkStart.x || g_ix >= chunkEndClamped.x ||
        g_iy < chunkStart.y || g_iy >= chunkEndClamped.y) {
        return;
    }

    if (s_isContourPixel[threadIdx.y][threadIdx.x] > 0) {
        if ((s_isContourPixel[threadIdx.y - 1][threadIdx.x] == 0) ||
            (s_isContourPixel[threadIdx.y][threadIdx.x - 1] == 0) ||
            (s_isContourPixel[threadIdx.y + 1][threadIdx.x] == 0) ||
            (s_isContourPixel[threadIdx.y][threadIdx.x + 1] == 0)) {

            contourBitmap[g_idx] = s_isContourPixel[threadIdx.y][threadIdx.x];
        }
    }
}

// ============================================================================
// Device Helper Functions
// ============================================================================

// ============================================================================
// Border-Following Algorithm (Suzuki-Abe style)
// ============================================================================
// This algorithm traces the BOUNDARY between foreground and background pixels,
// not the pixels themselves. It handles single-pixel-width protrusions correctly
// by tracing both sides of the protrusion as part of a continuous boundary.
//
// Coordinate System:
// - Vertex coordinates (vx, vy) are at grid corners, not pixel centers
// - Pixel (px, py) occupies the square with corners at:
//   (px, py), (px+1, py), (px, py+1), (px+1, py+1)
// - We trace along edges between vertices, keeping foreground on our right
//
// Direction encoding (for edge travel):
//   0 = right (+x), 1 = down (+y), 2 = left (-x), 3 = up (-y)
// ============================================================================

// Forward declaration (defined later in the file)
__device__ inline void addPolygonID(
    unsigned int* idBuffer,
    unsigned int* idCount,
    unsigned int newID,
    unsigned int maxIDs);

// Direction vectors for vertex movement
__device__ __constant__ int borderDvx[4] = {1, 0, -1, 0};  // right, down, left, up
__device__ __constant__ int borderDvy[4] = {0, 1, 0, -1};

// For each direction, the pixel that should be on our RIGHT side
// When at vertex (vx, vy) heading in direction d:
//   dir 0 (right): pixel (vx, vy) is to our right (below the edge)
//   dir 1 (down):  pixel (vx-1, vy) is to our right (left of edge)
//   dir 2 (left):  pixel (vx-1, vy-1) is to our right (above edge)
//   dir 3 (up):    pixel (vx, vy-1) is to our right (right of edge)
// Pixel to the right of the edge when traveling in direction dir from vertex (vx,vy)
// dir=0 (right): walking along top edge of pixel, foreground pixel is (vx, vy)
// dir=1 (down): walking along left edge of pixel, foreground pixel is (vx-1, vy)
// dir=2 (left): walking along bottom edge of pixel to top-left, foreground pixel is (vx-1, vy-1)
// dir=3 (up): walking along right edge of pixel to top-left, foreground pixel is (vx, vy-1)
__device__ __constant__ int rightPixelDx[4] = {0, -1, -1, 0};
__device__ __constant__ int rightPixelDy[4] = {0, 0, -1, -1};

// For each direction, the pixel that should be on our LEFT side
// dir 0 (right): pixel (vx, vy-1) is to our left (above the edge)
// dir 1 (down):  pixel (vx, vy) is to our left (right of edge)
// dir 2 (left):  pixel (vx-1, vy) is to our left (below edge)
// dir 3 (up):    pixel (vx-1, vy-1) is to our left (left of edge)
__device__ __constant__ int leftPixelDx[4] = {0, 0, -1, -1};
__device__ __constant__ int leftPixelDy[4] = {-1, 0, 0, -1};

/**
 * @brief Check if a pixel is foreground (has the target value) - for contour bitmap
 */
__device__ inline bool isForegroundPixel(
    const unsigned int* bitmap,
    int px, int py,
    unsigned int targetValue,
    unsigned int width,
    unsigned int height)
{
    if (px < 0 || px >= (int)width || py < 0 || py >= (int)height) {
        return false;
    }
    return bitmap[py * width + px] == targetValue;
}

/**
 * @brief Check if a pixel is foreground for INTERSECTION operation
 * Foreground = pixel has both subject (lower 16 bits) AND clipper (upper 16 bits) non-zero
 */
__device__ inline bool isIntersectionForegroundPixel(
    const unsigned int* overlayBitmap,
    int px, int py,
    unsigned int width,
    unsigned int height)
{
    if (px < 0 || px >= (int)width || py < 0 || py >= (int)height) {
        return false;
    }
    unsigned int overlayValue = overlayBitmap[py * width + px];
    unsigned int subjectId = overlayValue & 0xFFFF;
    unsigned int clipperId = (overlayValue >> 16) & 0xFFFF;

    // For intersection: foreground if BOTH subject and clipper are present
    return (subjectId != 0 && clipperId != 0);
}

/**
 * @brief Check if a pixel belongs to the SAME intersection region (same subject+clipper pair)
 * This is used for border tracing to ensure we stay within the same connected component
 */
__device__ inline bool isSameIntersectionRegion(
    const unsigned int* overlayBitmap,
    int px, int py,
    unsigned int targetValue,  // encoded as (clipper_id << 16) | subject_id
    unsigned int width,
    unsigned int height)
{
    if (px < 0 || px >= (int)width || py < 0 || py >= (int)height) {
        return false;
    }
    unsigned int overlayValue = overlayBitmap[py * width + px];

    // Check if this pixel has the same subject+clipper combination
    return (overlayValue == targetValue);
}

/**
 * @brief Get the pixel coordinates to the right of current edge direction
 * @param vx, vy Current vertex position
 * @param dir Current direction (0-3)
 * @param outPx, outPy Output pixel coordinates
 */
__device__ inline void getPixelToRight(
    int vx, int vy, int dir,
    int* outPx, int* outPy)
{
    *outPx = vx + rightPixelDx[dir];
    *outPy = vy + rightPixelDy[dir];
}

/**
 * @brief Get the pixel coordinates to the left of current edge direction
 * @param vx, vy Current vertex position
 * @param dir Current direction (0-3)
 * @param outPx, outPy Output pixel coordinates
 */
__device__ inline void getPixelToLeft(
    int vx, int vy, int dir,
    int* outPx, int* outPy)
{
    *outPx = vx + leftPixelDx[dir];
    *outPy = vy + leftPixelDy[dir];
}

/**
 * @brief Find a starting position for border tracing using overlay bitmap
 * Finds a foreground pixel (same intersection region) with background to its left,
 * returns the vertex position and direction for starting the trace.
 *
 * @param overlayBitmap The overlay bitmap (subject in low 16 bits, clipper in high 16 bits)
 * @param startPixelIdx The pixel index to start from (from sorted indices)
 * @param targetValue The target pixel value (subject+clipper encoded)
 * @param width, height Bitmap dimensions
 * @param outVx, outVy Output starting vertex position
 * @param outDir Output starting direction
 * @return true if valid start found, false otherwise
 */
__device__ bool findBorderStartPosition(
    const unsigned int* overlayBitmap,
    unsigned int startPixelIdx,
    unsigned int targetValue,
    unsigned int width,
    unsigned int height,
    int* outVx, int* outVy, int* outDir)
{
    int px = startPixelIdx % width;
    int py = startPixelIdx / width;

    // Verify this pixel belongs to the target intersection region
    if (!isSameIntersectionRegion(overlayBitmap, px, py, targetValue, width, height)) {
        return false;
    }

    // Try to find an edge where we have background (different region) on one side
    // Check each of the 4 edges of this pixel

    // Check left edge: if pixel to left is not same region, start here going UP
    if (!isSameIntersectionRegion(overlayBitmap, px - 1, py, targetValue, width, height)) {
        *outVx = px;      // Left edge of pixel
        *outVy = py + 1;  // Bottom of left edge
        *outDir = 3;      // Going up (foreground on right)
        return true;
    }

    // Check top edge: if pixel above is not same region, start here going right
    if (!isSameIntersectionRegion(overlayBitmap, px, py - 1, targetValue, width, height)) {
        *outVx = px;      // Left of top edge
        *outVy = py;      // Top edge of pixel
        *outDir = 0;      // Going right (foreground on right/below)
        return true;
    }

    // Check right edge: if pixel to right is not same region, start here going DOWN
    if (!isSameIntersectionRegion(overlayBitmap, px + 1, py, targetValue, width, height)) {
        *outVx = px + 1;  // Right edge of pixel
        *outVy = py;      // Top of right edge
        *outDir = 1;      // Going down (foreground on right)
        return true;
    }

    // Check bottom edge: if pixel below is not same region, start here going left
    if (!isSameIntersectionRegion(overlayBitmap, px, py + 1, targetValue, width, height)) {
        *outVx = px + 1;  // Right of bottom edge
        *outVy = py + 1;  // Bottom edge of pixel
        *outDir = 2;      // Going left (foreground on right/above)
        return true;
    }

    // This pixel is completely surrounded by same region - not a boundary pixel
    return false;
}

/**
 * @brief Trace one contour using border-following algorithm
 *
 * This traces the boundary between foreground and background pixels,
 * keeping foreground on the right side. Outputs vertex coordinates.
 */
__device__ void traceBorderContour(
    const unsigned int* contourBitmap,
    const unsigned int* overlayBitmap,
    unsigned int startPixelIdx,
    unsigned int targetValue,
    unsigned char* edgeVisited,  // Edge visited array: [vy * (width+1) * 4 + vx * 4 + dir]
    uint2* outputContour,
    unsigned int* outputCount,
    unsigned int* subjectIDs,
    unsigned int* clipperIDs,
    unsigned int* subjectCount,
    unsigned int* clipperCount,
    unsigned int width,
    unsigned int height,
    unsigned int maxPoints,
    unsigned int groupIdx,
    unsigned int contourIdx)
{
    *subjectCount = 0;
    *clipperCount = 0;
    *outputCount = 0;

    bool debug = (groupIdx == 1);  // Debug group 1 which should have more points

    int px = startPixelIdx % width;
    int py = startPixelIdx / width;

    if (debug) {
        printf("# [Border Group %u, Contour %u] Starting pixel idx=%u, pos=(%d,%d), targetValue=%u\n",
               groupIdx, contourIdx, startPixelIdx, px, py, targetValue);

        // Print the actual overlay value at this location
        unsigned int overlayValue = overlayBitmap[startPixelIdx];
        unsigned int subj = overlayValue & 0xFFFF;
        unsigned int clip = (overlayValue >> 16) & 0xFFFF;
        printf("#   Overlay value at pixel: %u (subj=%u, clip=%u)\n", overlayValue, subj, clip);

        // Check neighbors using overlay bitmap (same intersection region)
        printf("#   Neighbors sameRegion (L,R,U,D): ");
        printf("L=%d ", isSameIntersectionRegion(overlayBitmap, px-1, py, targetValue, width, height) ? 1 : 0);
        printf("R=%d ", isSameIntersectionRegion(overlayBitmap, px+1, py, targetValue, width, height) ? 1 : 0);
        printf("U=%d ", isSameIntersectionRegion(overlayBitmap, px, py-1, targetValue, width, height) ? 1 : 0);
        printf("D=%d ", isSameIntersectionRegion(overlayBitmap, px, py+1, targetValue, width, height) ? 1 : 0);
        printf("\n");
    }

    // Find starting position using overlay bitmap
    int startVx, startVy, startDir;
    if (!findBorderStartPosition(overlayBitmap, startPixelIdx, targetValue,
                                  width, height, &startVx, &startVy, &startDir)) {
        if (debug) {
            printf("# [Border Group %u, Contour %u] Could not find start position for pixel %u\n",
                   groupIdx, contourIdx, startPixelIdx);
        }
        return;
    }

    int vx = startVx;
    int vy = startVy;
    int dir = startDir;

    unsigned int pointCount = 0;
    unsigned int maxIter = (width + height) * 4 + 1000;  // Safety limit
    unsigned int iterCount = 0;

    // Edge visited index calculation helper
    unsigned int edgeWidth = width + 1;

    if (debug) {
        printf("# [Border Group %u, Contour %u] Starting border trace at vertex (%d, %d), dir=%d, targetValue=%u\n",
               groupIdx, contourIdx, startVx, startVy, startDir, targetValue);
        printf("border_contour_%u_%u = [\n", groupIdx, contourIdx);
    }

    // Mark starting edge as visited
    unsigned int startEdgeIdx = startVy * edgeWidth * 4 + startVx * 4 + startDir;
    edgeVisited[startEdgeIdx] = 1;

    do {
        // Output current vertex position
        if (pointCount < maxPoints) {
            outputContour[pointCount].x = (unsigned int)vx;
            outputContour[pointCount].y = (unsigned int)vy;

            if (debug) {
                printf("    (%d, %d),  # Point %u, dir=%d\n", vx, vy, pointCount, dir);
            }
            pointCount++;
        }

        // Collect polygon IDs from adjacent foreground pixel
        int adjPx, adjPy;
        getPixelToRight(vx, vy, dir, &adjPx, &adjPy);
        if (adjPx >= 0 && adjPx < (int)width && adjPy >= 0 && adjPy < (int)height) {
            unsigned int overlayValue = overlayBitmap[adjPy * width + adjPx];
            unsigned int subject_id = overlayValue & 0xFFFF;
            unsigned int clipper_id = (overlayValue >> 16) & 0xFFFF;
            addPolygonID(subjectIDs, subjectCount, subject_id, 256);
            addPolygonID(clipperIDs, clipperCount, clipper_id, 256);
        }

        // Try to turn right first (check if there's foreground in that direction AND background on left)
        int tryDir = (dir + 1) % 4;  // Turn right
        int newVx = vx + borderDvx[tryDir];
        int newVy = vy + borderDvy[tryDir];
        
        // Check pixel on right (should be FG)
        int checkPx, checkPy;
        getPixelToRight(vx, vy, tryDir, &checkPx, &checkPy);
        bool hasForeground = isSameIntersectionRegion(overlayBitmap, checkPx, checkPy, targetValue, width, height);

        // Check pixel on left (should be BG)
        int checkPxLeft, checkPyLeft;
        getPixelToLeft(vx, vy, tryDir, &checkPxLeft, &checkPyLeft);
        bool hasBackground = !isSameIntersectionRegion(overlayBitmap, checkPxLeft, checkPyLeft, targetValue, width, height);

        // Check if this edge has already been visited (to prevent infinite loops)
        // Edge is identified by (source_vertex, direction), so check from current position
        unsigned int tryEdgeIdx = vy * edgeWidth * 4 + vx * 4 + tryDir;
        bool edgeAlreadyVisited = (tryEdgeIdx < (height + 1) * edgeWidth * 4) && (edgeVisited[tryEdgeIdx] != 0);

        // Allow revisit only if it's the starting edge (to close the contour)
        // Starting edge is (startVx, startVy, startDir)
        bool isStartEdge = (vx == startVx && vy == startVy && tryDir == startDir);
        bool turnedRight = hasForeground && hasBackground && (!edgeAlreadyVisited || isStartEdge);

        if (debug) {
            printf("#   At vertex (%d,%d) dir=%d, try turn right to dir=%d, newV=(%d,%d), checkPixel=(%d,%d), isFG=%d, isBG=%d, visited=%d, isStart=%d\n",
                   vx, vy, dir, tryDir, newVx, newVy, checkPx, checkPy, hasForeground ? 1 : 0, hasBackground ? 1 : 0, edgeAlreadyVisited ? 1 : 0, isStartEdge ? 1 : 0);
        }

        // Save old position for edge marking
        int oldVx = vx;
        int oldVy = vy;
        int moveDir = dir;  // Direction we'll actually move in

        if (turnedRight) {
            // Can turn right - valid boundary edge and not visited
            moveDir = tryDir;
            dir = tryDir;
            vx = newVx;
            vy = newVy;
            if (debug) {
                printf("#     -> Turned right, new pos=(%d,%d), new dir=%d\n", vx, vy, dir);
            }
        } else {
            // Can't turn right - need to go straight or turn left
            // Keep turning left until we find a valid move
            int turnCount = 0;
            bool foundMove = false;
            while (turnCount < 4) {
                int nextVx = vx + borderDvx[dir];
                int nextVy = vy + borderDvy[dir];

                // Check pixel to the right of where we'd be going (must be FG)
                int checkPxStraight, checkPyStraight;
                getPixelToRight(vx, vy, dir, &checkPxStraight, &checkPyStraight);
                bool canGoForeground = isSameIntersectionRegion(overlayBitmap, checkPxStraight, checkPyStraight, targetValue, width, height);
                
                // Check pixel to the left (must be BG)
                int checkPxStraightLeft, checkPyStraightLeft;
                getPixelToLeft(vx, vy, dir, &checkPxStraightLeft, &checkPyStraightLeft);
                bool canGoBackground = !isSameIntersectionRegion(overlayBitmap, checkPxStraightLeft, checkPyStraightLeft, targetValue, width, height);

                // Check if this edge has already been visited
                // Edge is identified by (source_vertex, direction), so check from current position
                unsigned int straightEdgeIdx = vy * edgeWidth * 4 + vx * 4 + dir;
                bool straightEdgeVisited = (straightEdgeIdx < (height + 1) * edgeWidth * 4) && (edgeVisited[straightEdgeIdx] != 0);
                bool straightIsStart = (vx == startVx && vy == startVy && dir == startDir);
                bool canGoStraight = canGoForeground && canGoBackground && (!straightEdgeVisited || straightIsStart);

                if (debug) {
                    unsigned int actualVal = 0;
                    if (checkPxStraight >= 0 && checkPxStraight < (int)width &&
                        checkPyStraight >= 0 && checkPyStraight < (int)height) {
                        actualVal = overlayBitmap[checkPyStraight * width + checkPxStraight];
                    }
                    printf("#     Try straight: nextV=(%d,%d), checkPixel=(%d,%d), pixVal=%u (s=%u,c=%u), target=%u, isFG=%d, isBG=%d, visited=%d, isStart=%d, turnCount=%d\n",
                           nextVx, nextVy, checkPxStraight, checkPyStraight,
                           actualVal, actualVal & 0xFFFF, (actualVal >> 16) & 0xFFFF, targetValue,
                           canGoForeground ? 1 : 0, canGoBackground ? 1 : 0, straightEdgeVisited ? 1 : 0, straightIsStart ? 1 : 0, turnCount);
                }

                if (canGoStraight) {
                    // This edge borders foreground and not visited, follow it
                    moveDir = dir;  // We're moving in current direction
                    vx = nextVx;
                    vy = nextVy;
                    foundMove = true;
                    if (debug) {
                        printf("#     -> Went straight, new pos=(%d,%d)\n", vx, vy);
                    }
                    break;
                } else {
                    // Turn left (convex corner of foreground region)
                    dir = (dir + 3) % 4;
                    turnCount++;
                    if (debug) {
                        printf("#     -> Turn left, new dir=%d\n", dir);
                    }
                }
            }

            if (!foundMove) {
                // No valid move found - either isolated pixel or all edges visited
                if (debug) {
                    printf("]\n# [Border Group %u, Contour %u] No valid move found (turnCount=%d), ending trace\n\n",
                           groupIdx, contourIdx, turnCount);
                }
                break;
            }
        }

        // Mark the edge we just traversed as visited
        // Edge is identified by (starting_vertex, direction)
        unsigned int edgeIdx = oldVy * edgeWidth * 4 + oldVx * 4 + moveDir;
        if (edgeIdx < (height + 1) * edgeWidth * 4) {
            edgeVisited[edgeIdx] = 1;
        }

        iterCount++;

        // Debug: check loop condition
        if (debug) {
            bool atStart = (vx == startVx && vy == startVy && dir == startDir);
            printf("#   After move: pos=(%d,%d), dir=%d, atStart=%d, iterCount=%u\n",
                   vx, vy, dir, atStart ? 1 : 0, iterCount);
        }

    } while ((vx != startVx || vy != startVy || dir != startDir) && iterCount < maxIter);

    *outputCount = pointCount;

    if (debug) {
        if (vx == startVx && vy == startVy && dir == startDir) {
            printf("]\n# [Border Group %u, Contour %u] Closed contour with %u vertices\n\n",
                   groupIdx, contourIdx, pointCount);
        } else {
            printf("]\n# [Border Group %u, Contour %u] Max iterations reached (%u), trace incomplete\n\n",
                   groupIdx, contourIdx, iterCount);
        }
    }
}

// ============================================================================
// Original Pixel-Following Helper Functions (kept for backward compatibility)
// ============================================================================

__device__ inline unsigned int getBitmapPixel(
    const unsigned int* bitmap,
    int x, int y,
    unsigned int width,
    unsigned int height)
{
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0;
    }
    return bitmap[y * width + x];
}

__device__ inline int findSortedIndex(
    const unsigned int* sortedIndices,
    unsigned int targetIdx,
    unsigned int groupStart,
    unsigned int groupCount)
{
    for (unsigned int i = 0; i < groupCount; ++i) {
        if (sortedIndices[groupStart + i] == targetIdx) {
            return groupStart + i;
        }
    }
    return -1;
}

__device__ inline void addPolygonID(
    unsigned int* idBuffer,
    unsigned int* idCount,
    unsigned int newID,
    unsigned int maxIDs)
{
    if (newID == 0) return;

    for (unsigned int i = 0; i < *idCount; ++i) {
        if (idBuffer[i] == newID) {
            return;
        }
    }

    if (*idCount < maxIDs) {
        idBuffer[*idCount] = newID;
        (*idCount)++;
    }
}

// ============================================================================
// Device Function: Trace One Contour
// ============================================================================

__device__ void traceOneContour(
    const unsigned int* contourBitmap,
    const unsigned int* overlayBitmap,
    const unsigned int* sortedIndices,
    unsigned int startSortedIdx,
    unsigned int targetValue,
    unsigned int groupStart,
    unsigned int groupCount,
    unsigned char* visited,
    uint2* outputContour,
    unsigned int* outputCount,
    unsigned int* subjectIDs,
    unsigned int* clipperIDs,
    unsigned int* subjectCount,
    unsigned int* clipperCount,
    unsigned int width,
    unsigned int height,
    unsigned int maxPoints,
    unsigned int groupIdx,
    unsigned int contourIdx)
{
    unsigned int startIdx = sortedIndices[startSortedIdx];
    unsigned int startX = startIdx % width;
    unsigned int startY = startIdx / width;

    bool debug = (groupIdx == 11);

    *subjectCount = 0;
    *clipperCount = 0;

    const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    unsigned int currentX = startX;
    unsigned int currentY = startY;
    int currentDir = 0;

    unsigned int pointCount = 0;

    if (debug) {
        printf("# [Group %u, Contour %u] Starting trace at (%u, %u), targetValue=%u, startSortedIdx=%u\n",
               groupIdx, contourIdx, startX, startY, targetValue, startSortedIdx);
        printf("contour_%u_%u = [\n", groupIdx, contourIdx);
    }

    if (pointCount < maxPoints) {
        outputContour[pointCount].x = currentX;
        outputContour[pointCount].y = currentY;
        pointCount++;
        if (debug) {
            printf("    (%u, %u),  # Point %u | sortedIdx=%u, wasVisited=%d\n",
                   currentX, currentY, pointCount - 1, startSortedIdx,
                   visited[startSortedIdx] != 0 ? 1 : 0);
        }
    }

    unsigned int overlayValue = getBitmapPixel(overlayBitmap, currentX, currentY, width, height);
    unsigned int subject_id = overlayValue & 0xFFFF;
    unsigned int clipper_id = (overlayValue >> 16) & 0xFFFF;
    addPolygonID(subjectIDs, subjectCount, subject_id, 256);
    addPolygonID(clipperIDs, clipperCount, clipper_id, 256);

    visited[startSortedIdx] = 1;

    bool firstMove = true;
    unsigned int iterCount = 0;
    const unsigned int maxIter = width * height * 4;

    while (iterCount < maxIter) {
        iterCount++;

        int searchDir = (currentDir + 6) % 8;
        bool found = false;

        for (int i = 0; i < 8; ++i) {
            int checkDir = (searchDir + i) % 8;
            int nextX = currentX + dx[checkDir];
            int nextY = currentY + dy[checkDir];

            unsigned int pixelValue = getBitmapPixel(
                contourBitmap, nextX, nextY, width, height);

            if (pixelValue == targetValue) {
                unsigned int nextIdx = nextY * width + nextX;
                int sortedIdx = findSortedIndex(
                    sortedIndices, nextIdx, groupStart, groupCount);
                bool wasVisited = (sortedIdx >= 0) ? (visited[sortedIdx] != 0) : false;

                if (wasVisited && !(nextX == startX && nextY == startY)) {
                    if (debug) {
                        printf("# Skipping already-visited pixel (%d, %d) at direction %d\n",
                               nextX, nextY, checkDir);
                    }
                    continue;
                }

                currentX = nextX;
                currentY = nextY;
                currentDir = checkDir;
                found = true;

                if (!firstMove && currentX == startX && currentY == startY) {
                    *outputCount = pointCount;
                    if (debug) {
                        printf("]\n# [Group %u, Contour %u] Closed contour with %u points\n\n",
                               groupIdx, contourIdx, pointCount);
                    }
                    return;
                }
                firstMove = false;

                if (pointCount < maxPoints) {
                    outputContour[pointCount].x = currentX;
                    outputContour[pointCount].y = currentY;
                    if (debug) {
                        printf("    (%u, %u),  # Point %u | sortedIdx=%d, wasVisited=%d\n",
                               currentX, currentY, pointCount, sortedIdx, wasVisited ? 1 : 0);
                    }
                    pointCount++;
                }

                unsigned int overlayValue = getBitmapPixel(overlayBitmap, currentX, currentY, width, height);
                unsigned int subject_id = overlayValue & 0xFFFF;
                unsigned int clipper_id = (overlayValue >> 16) & 0xFFFF;
                addPolygonID(subjectIDs, subjectCount, subject_id, 256);
                addPolygonID(clipperIDs, clipperCount, clipper_id, 256);

                if (sortedIdx >= 0) {
                    visited[sortedIdx] = 1;
                }

                break;
            }
        }

        if (!found) {
            if (debug) {
                printf("]\n# [Group %u, Contour %u] No neighbor found, ending trace with %u points (OPEN contour)\n\n",
                       groupIdx, contourIdx, pointCount);
            }
            break;
        }
    }

    *outputCount = pointCount;

    if (debug && pointCount > 0 && iterCount >= maxIter) {
        printf("]\n# [Group %u, Contour %u] Max iterations reached, trace incomplete with %u points\n\n",
               groupIdx, contourIdx, pointCount);
    }
}

// ============================================================================
// Main Kernel: Each Block Processes One Group
// ============================================================================

__global__ void traceContoursParallel_kernel(
    const unsigned int* contourBitmap,
    const unsigned int* overlayBitmap,
    const unsigned int* sortedIndices,
    const unsigned int* sortedValues,
    const unsigned int* groupPixelValues,
    const unsigned int* groupStartIndices,
    const unsigned int* groupCounts,
    const unsigned int numGroups,
    unsigned char* visited,
    uint2* outputContours,
    unsigned int* outputCounts,
    unsigned int* outputSubjectIDs,
    unsigned int* outputClipperIDs,
    unsigned int* outputSubjectCounts,
    unsigned int* outputClipperCounts,
    const unsigned int width,
    const unsigned int height,
    const unsigned int maxPointsPerContour)
{
    unsigned int groupIdx = blockIdx.x;

    if (groupIdx >= numGroups) {
        return;
    }

    unsigned int targetValue = groupPixelValues[groupIdx];
    unsigned int groupStart = groupStartIndices[groupIdx];
    unsigned int groupCount = groupCounts[groupIdx];

    unsigned int contourCount = 0;
    const unsigned int maxContoursPerGroup = 32;
    const unsigned int maxIDsPerGroup = 256;

    __shared__ unsigned int sharedSubjectIDs[256];
    __shared__ unsigned int sharedClipperIDs[256];
    __shared__ unsigned int sharedSubjectCount;
    __shared__ unsigned int sharedClipperCount;

    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < groupCount && contourCount < maxContoursPerGroup; ++i) {
            unsigned int sortedIdx = groupStart + i;

            if (visited[sortedIdx] == 0) {
                unsigned int localCount = 0;
                uint2* contourOutput = outputContours + groupIdx * maxPointsPerContour;
                unsigned int currentOffset = outputCounts[groupIdx];

                unsigned int markerIdx = 0;
                if (contourCount > 0) {
                    if (currentOffset >= maxPointsPerContour - 1) {
                        break;
                    }
                    markerIdx = currentOffset;
                    contourOutput[currentOffset].x = 0xFFFFFFFF;
                    contourOutput[currentOffset].y = 0;
                    currentOffset++;
                    outputCounts[groupIdx]++;
                }

                traceOneContour(
                    contourBitmap,
                    overlayBitmap,
                    sortedIndices,
                    sortedIdx,
                    targetValue,
                    groupStart,
                    groupCount,
                    visited,
                    contourOutput + currentOffset,
                    &localCount,
                    sharedSubjectIDs,
                    sharedClipperIDs,
                    &sharedSubjectCount,
                    &sharedClipperCount,
                    width,
                    height,
                    maxPointsPerContour - currentOffset,
                    groupIdx,
                    contourCount);

                if (contourCount == 0) {
                    for (unsigned int j = 0; j < sharedSubjectCount && j < maxIDsPerGroup; ++j) {
                        outputSubjectIDs[groupIdx * maxIDsPerGroup + j] = sharedSubjectIDs[j];
                    }
                    for (unsigned int j = 0; j < sharedClipperCount && j < maxIDsPerGroup; ++j) {
                        outputClipperIDs[groupIdx * maxIDsPerGroup + j] = sharedClipperIDs[j];
                    }
                    outputSubjectCounts[groupIdx] = sharedSubjectCount;
                    outputClipperCounts[groupIdx] = sharedClipperCount;
                } else {
                    contourOutput[markerIdx].y = localCount;
                }

                outputCounts[groupIdx] += localCount;
                contourCount++;
            }
        }
    }
}

// ============================================================================
// Border-Following Kernel: Each Block Processes One Group
// ============================================================================
// This kernel uses the Suzuki-Abe style border-following algorithm that traces
// the boundary between foreground and background pixels. It correctly handles
// single-pixel-width protrusions by tracing both sides of the protrusion.
// ============================================================================

__global__ void traceContoursParallelBorder_kernel(
    const unsigned int* contourBitmap,
    const unsigned int* overlayBitmap,
    const unsigned int* sortedIndices,
    const unsigned int* sortedValues,
    const unsigned int* groupPixelValues,
    const unsigned int* groupStartIndices,
    const unsigned int* groupCounts,
    const unsigned int numGroups,
    unsigned char* edgeVisited,      // Edge visited array: size = (height+1) * (width+1) * 4
    unsigned char* pixelProcessed,   // Pixel processed array: size = width * height (to track starting pixels)
    uint2* outputContours,
    unsigned int* outputCounts,
    unsigned int* outputSubjectIDs,
    unsigned int* outputClipperIDs,
    unsigned int* outputSubjectCounts,
    unsigned int* outputClipperCounts,
    const unsigned int width,
    const unsigned int height,
    const unsigned int maxPointsPerContour)
{
    unsigned int groupIdx = blockIdx.x;

    if (groupIdx >= numGroups) {
        return;
    }

    unsigned int targetValue = groupPixelValues[groupIdx];
    unsigned int groupStart = groupStartIndices[groupIdx];
    unsigned int groupCount = groupCounts[groupIdx];

    unsigned int contourCount = 0;
    const unsigned int maxContoursPerGroup = 32;
    const unsigned int maxIDsPerGroup = 256;

    __shared__ unsigned int sharedSubjectIDs[256];
    __shared__ unsigned int sharedClipperIDs[256];
    __shared__ unsigned int sharedSubjectCount;
    __shared__ unsigned int sharedClipperCount;

    if (threadIdx.x == 0) {
        for (unsigned int i = 0; i < groupCount && contourCount < maxContoursPerGroup; ++i) {
            unsigned int sortedIdx = groupStart + i;
            unsigned int pixelIdx = sortedIndices[sortedIdx];

            // Skip if this pixel was already processed as part of another contour
            if (pixelProcessed[pixelIdx] != 0) {
                continue;
            }

            // Check if this pixel is on the boundary (has at least one neighbor not in same region)
            // Use overlay bitmap to check for same intersection region
            int px = pixelIdx % width;
            int py = pixelIdx / width;
            bool isBoundary = false;

            if (!isSameIntersectionRegion(overlayBitmap, px - 1, py, targetValue, width, height) ||
                !isSameIntersectionRegion(overlayBitmap, px + 1, py, targetValue, width, height) ||
                !isSameIntersectionRegion(overlayBitmap, px, py - 1, targetValue, width, height) ||
                !isSameIntersectionRegion(overlayBitmap, px, py + 1, targetValue, width, height)) {
                isBoundary = true;
            }

            if (!isBoundary) {
                // Interior pixel, mark as processed and skip
                pixelProcessed[pixelIdx] = 1;
                continue;
            }

            // This is a boundary pixel, trace the contour
            unsigned int localCount = 0;
            uint2* contourOutput = outputContours + groupIdx * maxPointsPerContour;
            unsigned int currentOffset = outputCounts[groupIdx];

            unsigned int markerIdx = 0;
            if (contourCount > 0) {
                if (currentOffset >= maxPointsPerContour - 1) {
                    break;
                }
                markerIdx = currentOffset;
                contourOutput[currentOffset].x = 0xFFFFFFFF;
                contourOutput[currentOffset].y = 0;
                currentOffset++;
                outputCounts[groupIdx]++;
            }

            traceBorderContour(
                contourBitmap,
                overlayBitmap,
                pixelIdx,
                targetValue,
                edgeVisited,
                contourOutput + currentOffset,
                &localCount,
                sharedSubjectIDs,
                sharedClipperIDs,
                &sharedSubjectCount,
                &sharedClipperCount,
                width,
                height,
                maxPointsPerContour - currentOffset,
                groupIdx,
                contourCount);

            // Mark all pixels in this group that are adjacent to the traced boundary
            // For simplicity, mark the starting pixel as processed
            pixelProcessed[pixelIdx] = 1;

            if (contourCount == 0) {
                for (unsigned int j = 0; j < sharedSubjectCount && j < maxIDsPerGroup; ++j) {
                    outputSubjectIDs[groupIdx * maxIDsPerGroup + j] = sharedSubjectIDs[j];
                }
                for (unsigned int j = 0; j < sharedClipperCount && j < maxIDsPerGroup; ++j) {
                    outputClipperIDs[groupIdx * maxIDsPerGroup + j] = sharedClipperIDs[j];
                }
                outputSubjectCounts[groupIdx] = sharedSubjectCount;
                outputClipperCounts[groupIdx] = sharedClipperCount;
            } else {
                contourOutput[markerIdx].y = localCount;
            }

            outputCounts[groupIdx] += localCount;
            contourCount++;

            // After tracing, mark all boundary pixels of this contour as processed
            // by scanning through the group and checking if they're on traced edges
            // (This is a simplified approach - mark all boundary pixels in group)
            for (unsigned int j = 0; j < groupCount; ++j) {
                unsigned int pIdx = sortedIndices[groupStart + j];
                int ppx = pIdx % width;
                int ppy = pIdx / width;

                // Check if this pixel has a neighbor not in same region (is boundary)
                // Use overlay bitmap for same intersection region check
                if (!isSameIntersectionRegion(overlayBitmap, ppx - 1, ppy, targetValue, width, height) ||
                    !isSameIntersectionRegion(overlayBitmap, ppx + 1, ppy, targetValue, width, height) ||
                    !isSameIntersectionRegion(overlayBitmap, ppx, ppy - 1, targetValue, width, height) ||
                    !isSameIntersectionRegion(overlayBitmap, ppx, ppy + 1, targetValue, width, height)) {
                    pixelProcessed[pIdx] = 1;
                }
            }
        }
    }
}

} // namespace GpuLithoLib
