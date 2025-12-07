#pragma once

#include "../include/gpuLitho.h"
#include "../include/operation_types.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace GpuLithoLib {

// Forward declarations
class LayerImpl;

using gpuLitho::OperationType;

// Structure to hold sorted contour pixel data
struct SortedContourPixels {
    thrust::device_vector<unsigned int> values;   // Sorted pixel values (keys)
    thrust::device_vector<unsigned int> indices;  // Sorted pixel indices (locations in bitmap)
    unsigned int count;                            // Number of non-zero pixels

    SortedContourPixels() : count(0) {}
};

/**
 * @brief Engine for detecting raw contours from bitmap data
 *
 * This engine handles:
 * - Extracting contour pixels from overlay bitmaps
 * - Sorting contour pixels by value for parallel processing
 * - Tracing contours using GPU parallel algorithms
 */
class RawContourDetectEngine {
public:
    RawContourDetectEngine();
    ~RawContourDetectEngine();

    /**
     * @brief Detect raw contours from output layer bitmap
     * @param outputLayer Layer with overlay bitmap
     * @param opType Operation type
     * @param currentGridWidth Grid width
     * @param currentGridHeight Grid height
     * @return Vector of raw contours
     */
    std::vector<std::vector<cv::Point>> detectRawContours(
        LayerImpl* outputLayer,
        OperationType opType,
        unsigned int currentGridWidth,
        unsigned int currentGridHeight);

    /**
     * @brief Sort contour pixels by value using Thrust library
     * This groups pixels with the same value together for parallel contour tracing
     *
     * @param contourBitmap Device pointer to contour bitmap
     * @param width Bitmap width
     * @param height Bitmap height
     * @return SortedContourPixels containing sorted values and indices
     */
    SortedContourPixels sortContourPixelsByValue(
        const unsigned int* contourBitmap,
        unsigned int width,
        unsigned int height);

    /**
     * @brief Trace contours using GPU parallel tracing for INTERSECTION operation
     * Each group (same pixel value) is processed by one GPU block
     *
     * @param sortedPixels Sorted pixel data from sortContourPixelsByValue
     * @param contourBitmap Device pointer to contour bitmap
     * @param overlayBitmap Device pointer to overlay bitmap (for polygon ID extraction)
     * @param width Bitmap width
     * @param height Bitmap height
     * @return Vector of contours as cv::Point vectors
     */
    std::vector<std::vector<cv::Point>> traceContoursGPU(
        const SortedContourPixels& sortedPixels,
        const unsigned int* contourBitmap,
        const unsigned int* overlayBitmap,
        unsigned int width,
        unsigned int height);
};

// ============================================================================
// GPU Kernel Declarations
// ============================================================================

/**
 * @brief Extract contour pixels from overlay bitmap
 */
__global__ void extractContours_kernel(
    const unsigned int* inputBitmap,
    unsigned int* contourBitmap,
    const int width,
    const int height,
    const int chunkDim,
    OperationType opType);

/**
 * @brief GPU kernel for parallel contour tracing using Suzuki-Abe algorithm
 * Each block processes one pixel value group (one contour component)
 */
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
    const unsigned int maxPointsPerContour);

} // namespace GpuLithoLib
