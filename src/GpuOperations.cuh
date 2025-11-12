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
 * @brief Contour extraction kernel using 4-connectivity
 */
__global__ void extractContours_kernel(
    const unsigned int* inputBitmap,
    unsigned int* contourBitmap,
    const int width,
    const int height,
    const int chunkDim,
    OperationType opType);

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
 */
__global__ void computeIntersectionPoints_kernel(
    const unsigned int* packedPairs,
    const unsigned int numPairs,
    const uint2* subjectVertices,
    const unsigned int* subjectStartIndices,
    const unsigned int* subjectPtCounts,
    const uint2* clipperVertices,
    const unsigned int* clipperStartIndices,
    const unsigned int* clipperPtCounts,
    IntersectionPointData* outputPoints,
    unsigned int* outputCounts);

} // namespace GpuLithoLib