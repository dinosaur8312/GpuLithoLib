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

// Device helper functions for intersection computation (inline to support cross-compilation unit calls)
__device__ __forceinline__ float calculateDistanceThreshold_device(float angle_degrees) {
    // Clamp angle to valid range
    angle_degrees = fmaxf(0.0f, fminf(90.0f, angle_degrees));

    // For perpendicular intersections (90 degrees), use minimum threshold
    if (angle_degrees >= 89.0f) {
        return 2.0f;
    }

    // For very acute angles (<= 1 degree), use maximum threshold
    if (angle_degrees <= 1.0f) {
        return 12.0f;
    }

    // Logarithmic interpolation for angles between 1 and 89 degrees
    float log_angle = logf(angle_degrees);
    float log_max = logf(89.0f);
    float normalized = 1.0f - (log_angle / log_max);

    float threshold = 2.0f + 10.0f * normalized;

    // Clamp to [2.0, 12.0] for safety
    return fmaxf(2.0f, fminf(12.0f, threshold));
}

__device__ __forceinline__ bool computeLineIntersection_device(
    float p1x, float p1y, float p2x, float p2y,
    float p3x, float p3y, float p4x, float p4y,
    float& intersectX, float& intersectY, float& angle_degrees) {

    float denom = (p1x - p2x) * (p3y - p4y) - (p1y - p2y) * (p3x - p4x);

    if (fabsf(denom) < EPSILON) {
        return false; // Lines are parallel
    }

    float t = ((p1x - p3x) * (p3y - p4y) - (p1y - p3y) * (p3x - p4x)) / denom;
    float u = -((p1x - p2x) * (p1y - p3y) - (p1y - p2y) * (p1x - p3x)) / denom;

    if (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f) {
        intersectX = p1x + t * (p2x - p1x);
        intersectY = p1y + t * (p2y - p1y);

        // Calculate angle between the two line segments
        float v1x = p2x - p1x;
        float v1y = p2y - p1y;
        float v2x = p4x - p3x;
        float v2y = p4y - p3y;

        float dot = v1x * v2x + v1y * v2y;
        float norm1 = sqrtf(v1x * v1x + v1y * v1y);
        float norm2 = sqrtf(v2x * v2x + v2y * v2y);

        if (norm1 > EPSILON && norm2 > EPSILON) {
            float cos_angle = dot / (norm1 * norm2);
            cos_angle = fmaxf(-1.0f, fminf(1.0f, cos_angle)); // Clamp to avoid NaN

            float angle_rad = acosf(fabsf(cos_angle));
            angle_degrees = angle_rad * 180.0f / M_PI;

            // Ensure we get the acute angle
            if (angle_degrees > 90.0f) {
                angle_degrees = 180.0f - angle_degrees;
            }
        }

        return true;
    }

    return false;
}

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