#include "RawContourDetectEngine.cuh"
#include "GpuKernelProfiler.cuh"
#include "LayerImpl.h"
#include "GpuOperations.cuh"
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

    // Save binary image for debugging
    std::string debugFilename = "debug_binary_";
    switch(opType) {
        case OperationType::OFFSET:
            debugFilename += "offset.png";
            break;
        case OperationType::INTERSECTION:
            debugFilename += "intersection.png";
            break;
        case OperationType::UNION:
            debugFilename += "union.png";
            break;
        case OperationType::DIFFERENCE:
            debugFilename += "difference.png";
            break;
        case OperationType::XOR:
            debugFilename += "xor.png";
            break;
        default:
            debugFilename += "unknown.png";
            break;
    }
    cv::imwrite(debugFilename, binaryImage);

    // Use OpenCV to find contours
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);

    // ========================================================================
    // Save CPU and GPU raw contours separately
    // ========================================================================
    if (opType == OperationType::INTERSECTION) {
        // Save CPU raw contours
        cv::Mat cpuRawImage = cv::Mat::zeros(currentGridHeight, currentGridWidth, CV_8UC3);
        cv::Scalar cpuColor(0, 255, 0);  // Green
        for (size_t i = 0; i < contours.size(); ++i) {
            if (contours[i].size() > 1) {
                cv::polylines(cpuRawImage, contours[i], true, cpuColor, 1, cv::LINE_8);
            }
        }
        cv::imwrite("cpu_raw_contours.png", cpuRawImage);
        std::cout << "Saved CPU raw contours: cpu_raw_contours.png (" << contours.size() << " contours)" << std::endl;

        // Save GPU raw contours
        if (!gpuContours.empty()) {
            cv::Mat gpuRawImage = cv::Mat::zeros(currentGridHeight, currentGridWidth, CV_8UC3);
            cv::Scalar gpuColor(0, 0, 255);  // Red
            for (size_t i = 0; i < gpuContours.size(); ++i) {
                if (gpuContours[i].size() > 1) {
                    cv::polylines(gpuRawImage, gpuContours[i], true, gpuColor, 1, cv::LINE_8);
                }
            }
            cv::imwrite("gpu_raw_contours.png", gpuRawImage);
            std::cout << "Saved GPU raw contours: gpu_raw_contours.png (" << gpuContours.size() << " contours)" << std::endl;
        }
    }

    // ========================================================================
    // Comparison Visualization for INTERSECTION operation
    // ========================================================================
    if (opType == OperationType::INTERSECTION && !gpuContours.empty()) {
        std::cout << "Creating comparison visualization: GPU vs OpenCV contours" << std::endl;
        std::cout << "  GPU contours: " << gpuContours.size() << std::endl;
        std::cout << "  OpenCV contours: " << contours.size() << std::endl;

        // Create RGB image for comparison
        cv::Mat comparisonImage = cv::Mat::zeros(currentGridHeight, currentGridWidth, CV_8UC3);

        // Draw OpenCV contours in GREEN using polylines
        cv::Scalar greenColor(0, 255, 0);
        for (size_t i = 0; i < contours.size(); ++i) {
            if (contours[i].size() > 1) {
                cv::polylines(comparisonImage, contours[i], true, greenColor, 1, cv::LINE_8);
            }
        }

        // Draw GPU contours in RED using polylines (overlay on top)
        cv::Scalar redColor(0, 0, 255);
        for (size_t i = 0; i < gpuContours.size(); ++i) {
            if (gpuContours[i].size() > 1) {
                cv::polylines(comparisonImage, gpuContours[i], true, redColor, 1, cv::LINE_8);
            }
        }

        // Save comparison image (CPU first, GPU on top)
        std::string comparisonFilename = "contour_comparison_intersection_cpu_gpu.png";
        cv::imwrite(comparisonFilename, comparisonImage);
        std::cout << "  Saved comparison to: " << comparisonFilename << std::endl;
        std::cout << "  Color legend: GREEN=OpenCV contours (bottom), RED=GPU contours (overlay on top)" << std::endl;

        // Create reverse order comparison image (GPU first, CPU on top)
        cv::Mat reverseComparisonImage = cv::Mat::zeros(currentGridHeight, currentGridWidth, CV_8UC3);

        // Draw GPU contours in RED first (bottom layer)
        for (size_t i = 0; i < gpuContours.size(); ++i) {
            if (gpuContours[i].size() > 1) {
                cv::polylines(reverseComparisonImage, gpuContours[i], true, redColor, 1, cv::LINE_8);
            }
        }

        // Draw OpenCV contours in GREEN on top (overlay)
        for (size_t i = 0; i < contours.size(); ++i) {
            if (contours[i].size() > 1) {
                cv::polylines(reverseComparisonImage, contours[i], true, greenColor, 1, cv::LINE_8);
            }
        }

        // Save reverse order comparison image
        std::string reverseComparisonFilename = "contour_comparison_intersection_gpu_cpu.png";
        cv::imwrite(reverseComparisonFilename, reverseComparisonImage);
        std::cout << "  Saved reverse comparison to: " << reverseComparisonFilename << std::endl;

        return contours;
    }

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

    // Step 3: Allocate visited array
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

    // ========== DEBUG: Visualize Group 11 contour pixels ==========
    const unsigned int DEBUG_GROUP = 11;
    if (DEBUG_GROUP < numGroups) {
        thrust::host_vector<unsigned int> h_group_starts = d_group_starts;
        thrust::host_vector<unsigned int> h_group_counts = d_group_counts;
        thrust::host_vector<unsigned int> h_unique_values = d_unique_values;
        thrust::host_vector<unsigned int> h_sortedIndices = sortedPixels.indices;

        unsigned int groupStart = h_group_starts[DEBUG_GROUP];
        unsigned int groupCount = h_group_counts[DEBUG_GROUP];
        unsigned int groupValue = h_unique_values[DEBUG_GROUP];

        std::cout << "DEBUG Group " << DEBUG_GROUP << ": value=" << groupValue
                  << ", start=" << groupStart << ", count=" << groupCount << std::endl;

        // Find bounding box of group 11 pixels
        unsigned int minX = width, maxX = 0, minY = height, maxY = 0;
        for (unsigned int i = 0; i < groupCount; ++i) {
            unsigned int pixelIdx = h_sortedIndices[groupStart + i];
            unsigned int px = pixelIdx % width;
            unsigned int py = pixelIdx / width;
            minX = std::min(minX, px);
            maxX = std::max(maxX, px);
            minY = std::min(minY, py);
            maxY = std::max(maxY, py);
        }

        // Add margin for better visualization
        unsigned int margin = 5;
        minX = (minX > margin) ? (minX - margin) : 0;
        minY = (minY > margin) ? (minY - margin) : 0;
        maxX = std::min(maxX + margin, width - 1);
        maxY = std::min(maxY + margin, height - 1);

        unsigned int regionWidth = maxX - minX + 1;
        unsigned int regionHeight = maxY - minY + 1;

        std::cout << "  Bounding box: (" << minX << "," << minY << ") to ("
                  << maxX << "," << maxY << ")" << std::endl;

        // Create visualization image (scale up for better visibility)
        unsigned int scale = 10;
        cv::Mat visImage = cv::Mat::zeros(regionHeight * scale, regionWidth * scale, CV_8UC3);

        // Draw all pixels in group 11
        for (unsigned int i = 0; i < groupCount; ++i) {
            unsigned int pixelIdx = h_sortedIndices[groupStart + i];
            unsigned int px = pixelIdx % width;
            unsigned int py = pixelIdx / width;

            if (px >= minX && px <= maxX && py >= minY && py <= maxY) {
                unsigned int imgX = (px - minX) * scale;
                unsigned int imgY = (py - minY) * scale;
                cv::rectangle(visImage,
                            cv::Point(imgX, imgY),
                            cv::Point(imgX + scale - 1, imgY + scale - 1),
                            cv::Scalar(255, 255, 255), cv::FILLED);
            }
        }

        // Draw grid lines and axis labels
        for (unsigned int x = 0; x <= regionWidth; ++x) {
            int imgX = x * scale;
            cv::line(visImage, cv::Point(imgX, 0), cv::Point(imgX, regionHeight * scale - 1),
                    cv::Scalar(50, 50, 50), 1);

            if (x % 5 == 0 && x < regionWidth) {
                std::string label = std::to_string(minX + x);
                cv::putText(visImage, label, cv::Point(imgX + 2, 12),
                           cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
            }
        }
        for (unsigned int y = 0; y <= regionHeight; ++y) {
            int imgY = y * scale;
            cv::line(visImage, cv::Point(0, imgY), cv::Point(regionWidth * scale - 1, imgY),
                    cv::Scalar(50, 50, 50), 1);

            if (y % 5 == 0 && y < regionHeight) {
                std::string label = std::to_string(minY + y);
                cv::putText(visImage, label, cv::Point(2, imgY + 12),
                           cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
            }
        }

        std::string filename = "debug_group11_contour_pixels.png";
        cv::imwrite(filename, visImage);
        std::cout << "  Saved contour pixel visualization: " << filename << std::endl;
    }

    // Step 5: Launch tracing kernel
    dim3 blockSize(1, 1, 1);
    dim3 gridSize(numGroups, 1, 1);

    gpuEvent_t traceStart, traceStop;
    gpuEventCreate(&traceStart);
    gpuEventCreate(&traceStop);
    gpuEventRecord(traceStart);

    traceContoursParallel_kernel<<<gridSize, blockSize>>>(
        contourBitmap,
        overlayBitmap,
        thrust::raw_pointer_cast(sortedPixels.indices.data()),
        thrust::raw_pointer_cast(sortedPixels.values.data()),
        thrust::raw_pointer_cast(d_unique_values.data()),
        thrust::raw_pointer_cast(d_group_starts.data()),
        thrust::raw_pointer_cast(d_group_counts.data()),
        numGroups,
        thrust::raw_pointer_cast(d_visited.data()),
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

    // ========== VISUALIZATION ==========
    cv::Mat visImage = cv::Mat::zeros(height, width, CV_8UC3);
    thrust::host_vector<unsigned int> h_unique_values(d_unique_values);

    unsigned int contourIdx = 0;
    for (unsigned int g = 0; g < numGroups; ++g) {
        unsigned int totalCount = h_outputCounts[g];
        if (totalCount > 0) {
            unsigned int baseIdx = g * maxPointsPerContour;
            unsigned int i = 0;
            unsigned int contourInGroup = 0;

            cv::Scalar color;
            if (g < 100) {
                color = cv::Scalar(0, 0, 255);
            } else if (g < 200) {
                color = cv::Scalar(0, 255, 0);
            } else {
                color = cv::Scalar(255, 0, 0);
            }

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
                    for (unsigned int j = 0; j < contourLen; ++j) {
                        uint2 pt1 = h_outputContours[baseIdx + startIdx + j];
                        uint2 pt2 = h_outputContours[baseIdx + startIdx + ((j + 1) % contourLen)];

                        cv::Point p1(pt1.x, pt1.y);
                        cv::Point p2(pt2.x, pt2.y);
                        cv::line(visImage, p1, p2, color, 1, cv::LINE_AA);
                    }

                    unsigned int groupValue = h_unique_values[g];
                    unsigned int subject_id = (groupValue & 0xFFFF) - 1;
                    unsigned int clipper_id = ((groupValue >> 16) & 0xFFFF) - 1;

                    uint2 firstPt = h_outputContours[baseIdx + startIdx];
                    cv::Point labelPos(firstPt.x + 5, firstPt.y - 5);

                    std::string label = "(" + std::to_string(subject_id) + "," + std::to_string(clipper_id) + ")";

                    cv::putText(visImage, label, labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.3,
                                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

                    contourIdx++;
                }

                contourInGroup++;
            }
        }
    }

    std::string outputFilename = "gpu_contour_tracing_visualization.png";
    cv::imwrite(outputFilename, visImage);
    std::cout << "Saved GPU contour visualization to: " << outputFilename << std::endl;

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

} // namespace GpuLithoLib
