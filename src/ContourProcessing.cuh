#pragma once

#include "../include/gpuLitho.h"
#include "../include/operation_types.h"
#include "IntersectionCompute.cuh"
#include <vector>
#include <map>
#include <set>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

// Forward declarations
class LayerImpl;

// GPU Kernel Profiling Functions
void addRayCastingTime(float ms);
void addOverlayTime(float ms);
void addIntersectionComputeTime(float ms);
void printGpuKernelTimingSummary();
void resetGpuKernelTimers();

using gpuLitho::OperationType;

// Structure for candidate points in contour simplification
struct CandidatePoint {
    unsigned int x;
    unsigned int y;
    float max_distance_threshold;

    CandidatePoint(unsigned int px, unsigned int py, float threshold)
        : x(px), y(py), max_distance_threshold(threshold) {}
};

// Structure to hold sorted contour pixel data
struct SortedContourPixels {
    thrust::device_vector<unsigned int> values;   // Sorted pixel values (keys)
    thrust::device_vector<unsigned int> indices;  // Sorted pixel indices (locations in bitmap)
    unsigned int count;                            // Number of non-zero pixels

    SortedContourPixels() : count(0) {}
};

// Structure to hold a single contour point during tracing
struct ContourPoint {
    unsigned int x;
    unsigned int y;

    __host__ __device__ ContourPoint() : x(0), y(0) {}
    __host__ __device__ ContourPoint(unsigned int px, unsigned int py) : x(px), y(py) {}
};

// Note: Group data uses separate pointer representation for clarity:
// - d_group_pixel_values: Pixel value for each group
// - d_group_start_indices: Starting index in sorted arrays for each group
// - d_group_counts: Number of pixels in each group

// Note: Polygon ID data uses separate pointer representation for clarity:
// - d_subject_ids: Buffer for unique subject IDs [numGroups * 256]
// - d_clipper_ids: Buffer for unique clipper IDs [numGroups * 256]
// - d_subject_counts: Number of unique subject IDs per group [numGroups]
// - d_clipper_counts: Number of unique clipper IDs per group [numGroups]

// ContourDetectEngine: Manages contour detection and simplification
class ContourDetectEngine {
public:
    ContourDetectEngine();
    ~ContourDetectEngine();

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
     * @brief Simplify contours using layer vertex information and intersection points
     * @param raw_contours Raw contours from detectRawContours
     * @param subjectLayer Subject polygon layer
     * @param clipperLayer Clipper polygon layer
     * @param outputLayer Output layer with overlay bitmap
     * @param intersection_points_set Map of polygon pairs to intersection points
     * @param opType Operation type
     * @param currentGridWidth Grid width
     * @param currentGridHeight Grid height
     * @return Simplified contours
     */
    std::vector<std::vector<cv::Point>> simplifyContoursWithGeometry(
        const std::vector<std::vector<cv::Point>>& raw_contours,
        LayerImpl* subjectLayer,
        LayerImpl* clipperLayer,
        LayerImpl* outputLayer,
        const std::map<std::pair<unsigned int, unsigned int>, std::set<IntersectionPoint>>& intersection_points_set,
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
// GPU Kernel Declarations (implementations in ContourProcessing.cu)
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
    ContourPoint* outputContours,
    unsigned int* outputCounts,
    unsigned int* outputSubjectIDs,
    unsigned int* outputClipperIDs,
    unsigned int* outputSubjectCounts,
    unsigned int* outputClipperCounts,
    const unsigned int width,
    const unsigned int height,
    const unsigned int maxPointsPerContour);

} // namespace GpuLithoLib
