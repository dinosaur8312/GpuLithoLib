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

} // namespace GpuLithoLib