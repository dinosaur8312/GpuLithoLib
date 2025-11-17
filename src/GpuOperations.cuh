#pragma once

#include "../include/gpuLitho.h"
#include "../include/operation_types.h"

namespace GpuLithoLib {

using gpuLitho::OperationType;

// GPU kernels for lithography operations (simplified from original)

/**
 * @brief Ray casting kernel - fills polygon interiors with polygon ID
 */
__global__ void rayCasting_kernel(
    const uint2* vertices,
    const unsigned int* startIndices,
    const unsigned int* ptCounts,
    const uint4* boxes,
    unsigned int* bitmap,
    const int bitmapWidth,
    const int bitmapHeight,
    const unsigned int polygonNum);

/**
 * @brief Edge rendering kernel - renders polygon edges
 */
__global__ void edgeRender_kernel(
    const uint2* vertices,
    const unsigned int* startIndices,
    const unsigned int* ptCounts,
    unsigned int* bitmap,
    const int bitmapWidth,
    const int bitmapHeight,
    const int mode);

/**
 * @brief Overlay kernel - combines two bitmaps with polygon IDs
 */
__global__ void overlay_kernel(
    const unsigned int* subjectBitmap,
    const unsigned int* clipperBitmap,
    unsigned int* outputBitmap,
    int width,
    int height);

/**
 * @brief Offset kernel using edge convolution
 */
__global__ void offset_kernel(
    const uint2* vertices,
    const unsigned int* startIndices,
    const unsigned int* ptCounts,
    unsigned int* bitmap,
    const int bitmapWidth,
    const int bitmapHeight,
    const int offsetDistance,
    const bool positiveOffset);

/**
 * @brief Structure to store intersection point data (x, y, distanceThreshold)
 */
struct IntersectionPointData {
    unsigned int x;
    unsigned int y;
    float distanceThreshold;
};

// Device helper functions for intersection computation
__device__ float calculateDistanceThreshold_device(float angle_degrees);
__device__ bool computeLineIntersection_device(
    float p1x, float p1y, float p2x, float p2y,
    float p3x, float p3y, float p4x, float p4y,
    float& intersectX, float& intersectY, float& angle_degrees);

/**
 * @brief Compute intersection points for polygon pairs on GPU
 * Each block handles one intersecting pair
 * Assumes: 1) polygons have <= 1024 vertices, 2) max 32 intersection points per pair
 *
 * @param packedPairs Array of packed polygon pairs (clipper_id << 16 | subject_id)
 * @param numPairs Number of intersecting pairs
 * @param subjectVertices Subject layer vertices (device)
 * @param subjectStartIndices Subject layer start indices (device)
 * @param subjectPtCounts Subject layer vertex counts (device)
 * @param clipperVertices Clipper layer vertices (device)
 * @param clipperStartIndices Clipper layer start indices (device)
 * @param clipperPtCounts Clipper layer vertex counts (device)
 * @param outputPoints 2D array [numPairs][32] for intersection points (device)
 * @param outputCounts Array [numPairs] for actual count per pair (device)
 *
 * NOTE: Kernel declarations moved to appropriate files:
 * - computeIntersectionPoints_kernel -> IntersectionCompute.cuh
 * - extractContours_kernel -> ContourProcessing.cuh
 * - traceContoursParallel_kernel -> ContourProcessing.cuh
 */

} // namespace GpuLithoLib