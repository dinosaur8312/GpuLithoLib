#include "IntersectionCompute.cuh"
#include "GpuKernelProfiler.cuh"
#include "LayerImpl.h"
#include "CommonRenderUtils.cuh"
#include "../include/GpuLithoLib.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <cmath>
#include <iostream>

namespace GpuLithoLib {

// External profiler instance (owned by GpuLithoEngine)
extern GpuKernelProfiler* g_kernelProfiler;

// ============================================================================
// IntersectionComputeEngine Implementation
// ============================================================================

IntersectionComputeEngine::IntersectionComputeEngine()
    : d_packed_pairs(nullptr), d_intersection_points(nullptr),
      d_intersection_counts(nullptr), numIntersectingPairs(0) {}

IntersectionComputeEngine::~IntersectionComputeEngine() {
    freeData();
}

void IntersectionComputeEngine::freeData() {
    if (d_packed_pairs) {
        CHECK_GPU_ERROR(gpuFree(d_packed_pairs));
        d_packed_pairs = nullptr;
    }
    if (d_intersection_points) {
        CHECK_GPU_ERROR(gpuFree(d_intersection_points));
        d_intersection_points = nullptr;
    }
    if (d_intersection_counts) {
        CHECK_GPU_ERROR(gpuFree(d_intersection_counts));
        d_intersection_counts = nullptr;
    }
    numIntersectingPairs = 0;
}

// ============================================================================
// Helper Functions
// ============================================================================

// Helper function to calculate distance threshold based on intersection angle
// Matches the original algorithm from bitmap_layer.cu
float calculateDistanceThreshold(double angle_degrees) {
    // Clamp angle to valid range
    angle_degrees = std::max(0.0, std::min(90.0, angle_degrees));

    // For perpendicular intersections (90 degrees), use minimum threshold
    if (angle_degrees >= 89.0) {
        return 2.0f;
    }

    // For very acute angles (<= 1 degree), use maximum threshold
    if (angle_degrees <= 1.0) {
        return 10.0f;
    }

    // Logarithmic interpolation for angles between 1 and 89 degrees
    // As angle decreases, threshold increases
    double log_angle = std::log(angle_degrees);
    double log_max = std::log(89.0);
    double normalized = 1.0 - (log_angle / log_max);

    float threshold = 2.0f + 10.0f * static_cast<float>(normalized);

    // Clamp to [2.0, 12.0] for safety
    return std::max(2.0f, std::min(12.0f, threshold));
}

// Helper function to compute intersection between two line segments
// Returns true if intersection found, fills intersection point and angle
bool computeLineIntersection(const cv::Point2f& p1, const cv::Point2f& p2,
                             const cv::Point2f& p3, const cv::Point2f& p4,
                             cv::Point2f& intersection, double& angle_degrees) {

    float denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);

    if (std::abs(denom) < 1e-6) {
        return false; // Lines are parallel
    }

    float t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom;
    float u = -((p1.x - p2.x) * (p1.y - p3.y) - (p1.y - p2.y) * (p1.x - p3.x)) / denom;

    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        intersection.x = p1.x + t * (p2.x - p1.x);
        intersection.y = p1.y + t * (p2.y - p1.y);

        // Calculate angle between the two line segments
        cv::Point2f v1 = p2 - p1;
        cv::Point2f v2 = p4 - p3;

        float dot = v1.x * v2.x + v1.y * v2.y;
        float norm1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
        float norm2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);

        if (norm1 > 1e-6 && norm2 > 1e-6) {
            float cos_angle = dot / (norm1 * norm2);
            cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle)); // Clamp to avoid NaN

            double angle_rad = std::acos(std::abs(cos_angle));
            angle_degrees = angle_rad * 180.0 / M_PI;

            // Ensure we get the acute angle
            if (angle_degrees > 90.0) {
                angle_degrees = 180.0 - angle_degrees;
            }
        }

        return true;
    }

    return false;
}

// Compute all intersection points from output layer (combines extractIntersectingPolygonPairs + computeIntersectionPoints)
// Input: outputLayer after overlay operation
// Output: map of polygon pairs to their intersection points
// Stores device-side data in this engine for later use
std::map<std::pair<unsigned int, unsigned int>, std::set<IntersectionPoint>>
IntersectionComputeEngine::computeAllIntersectionPoints(
    LayerImpl* outputLayer,
    LayerImpl* subjectLayer,
    LayerImpl* clipperLayer,
    unsigned int currentGridWidth,
    unsigned int currentGridHeight) {

    std::map<std::pair<unsigned int, unsigned int>, std::set<IntersectionPoint>> intersection_points_set;

    if (!outputLayer || !subjectLayer || !clipperLayer) {
        std::cerr << "Error: Null parameter in computeAllIntersectionPoints" << std::endl;
        return intersection_points_set;
    }

    // Free any previous intersection data
    freeData();

    // Step 1: Extract intersecting polygon pairs from overlay bitmap using GPU
    // This is much more efficient than CPU loop + memcpy
    std::set<std::pair<unsigned int, unsigned int>> intersecting_pairs;

    unsigned int totalPixels = currentGridWidth * currentGridHeight;
    if (totalPixels == 0 || !outputLayer->d_bitmap) {
        std::cerr << "Warning: No bitmap data available for intersection extraction" << std::endl;
        return intersection_points_set;
    }

    // Create device vector view of the bitmap
    thrust::device_ptr<unsigned int> d_bitmap_ptr(outputLayer->d_bitmap);

    // Step 1a: Use thrust::copy_if to compact only intersection pixels
    // The bitmap already contains packed pairs: clipper_id (upper 16 bits) | subject_id (lower 16 bits)
    // We just need to filter pixels where both IDs are non-zero
    unsigned int* d_temp_packed_pairs;
    CHECK_GPU_ERROR(gpuMalloc(&d_temp_packed_pairs, totalPixels * sizeof(unsigned int)));
    thrust::device_ptr<unsigned int> d_temp_ptr(d_temp_packed_pairs);

    auto compact_end = thrust::copy_if(
        d_bitmap_ptr,                    // First iterator: start of bitmap
        d_bitmap_ptr + totalPixels,      // Last iterator: end of bitmap
        d_temp_ptr,                      // Output destination
        IsIntersectionPixel()            // Predicate functor
    );

    // Get compacted size
    auto compact_size = static_cast<size_t>(compact_end - d_temp_ptr);

    // Step 1b: Sort using radix sort (optimal for GPUs)
    thrust::sort(d_temp_ptr, d_temp_ptr + compact_size);

    // Step 1c: Remove duplicates to get unique pairs
    auto unique_end = thrust::unique(d_temp_ptr, d_temp_ptr + compact_size);
    auto unique_size = static_cast<size_t>(unique_end - d_temp_ptr);

    // Allocate persistent device memory (only for unique pairs)
    numIntersectingPairs = unique_size;
    CHECK_GPU_ERROR(gpuMalloc(&d_packed_pairs, unique_size * sizeof(unsigned int)));
    CHECK_GPU_ERROR(gpuMemcpy(d_packed_pairs, d_temp_packed_pairs,
                               unique_size * sizeof(unsigned int), gpuMemcpyDeviceToDevice));

    // Free temporary buffer
    CHECK_GPU_ERROR(gpuFree(d_temp_packed_pairs));

    // Step 1d: Copy unique pairs back to host (minimal transfer - only unique pairs)
    std::vector<unsigned int> h_packed_pairs(unique_size);
    CHECK_GPU_ERROR(gpuMemcpy(h_packed_pairs.data(), d_packed_pairs,
                               unique_size * sizeof(unsigned int), gpuMemcpyDeviceToHost));


    // Step 2: Compute intersection points for each intersecting polygon pair using GPU
    const unsigned int MAX_INTERSECTIONS_PER_PAIR = 64;  // Increased to handle endpoint cases
    unsigned int numPairs = numIntersectingPairs;

    if (numPairs > 0) {
        // Allocate persistent device memory for intersection points output
        // 2D array: [numPairs][32] intersection points
        CHECK_GPU_ERROR(gpuMalloc(&d_intersection_points,
                                   numPairs * MAX_INTERSECTIONS_PER_PAIR * sizeof(IntersectionPointData)));
        CHECK_GPU_ERROR(gpuMalloc(&d_intersection_counts, numPairs * sizeof(unsigned int)));

        // Launch kernel: one block per intersecting pair
        dim3 blockSize(256);  // 256 threads per block
        dim3 gridSize(numPairs);

        // Time the intersection computation kernel
        gpuEvent_t intStart, intStop;
        gpuEventCreate(&intStart);
        gpuEventCreate(&intStop);
        gpuEventRecord(intStart);

        computeIntersectionPoints_kernel<<<gridSize, blockSize>>>(
            d_packed_pairs,
            numPairs,
            subjectLayer->d_vertices,
            subjectLayer->d_startIndices,
            subjectLayer->d_ptCounts,
            clipperLayer->d_vertices,
            clipperLayer->d_startIndices,
            clipperLayer->d_ptCounts,
            d_intersection_points,
            d_intersection_counts);

        CHECK_GPU_ERROR(gpuGetLastError());
        gpuEventRecord(intStop);
        gpuEventSynchronize(intStop);

        float intMs = 0.0f;
        gpuEventElapsedTime(&intMs, intStart, intStop);
        if (g_kernelProfiler) g_kernelProfiler->addIntersectionComputeTime(intMs);

        gpuEventDestroy(intStart);
        gpuEventDestroy(intStop);

        // Copy results back to host for constructing the map
        std::vector<IntersectionPointData> h_intersection_points(numPairs * MAX_INTERSECTIONS_PER_PAIR);
        std::vector<unsigned int> h_intersection_counts(numPairs);

        CHECK_GPU_ERROR(gpuMemcpy(h_intersection_points.data(), d_intersection_points,
                                   numPairs * MAX_INTERSECTIONS_PER_PAIR * sizeof(IntersectionPointData),
                                   gpuMemcpyDeviceToHost));
        CHECK_GPU_ERROR(gpuMemcpy(h_intersection_counts.data(), d_intersection_counts,
                                   numPairs * sizeof(unsigned int), gpuMemcpyDeviceToHost));

        // Convert results to the map structure
        for (unsigned int pairIdx = 0; pairIdx < numPairs; ++pairIdx) {
            unsigned int packed_pair = h_packed_pairs[pairIdx];
            unsigned int subject_id = packed_pair & 0xFFFF;
            unsigned int clipper_id = (packed_pair >> 16) & 0xFFFF;

            unsigned int count = h_intersection_counts[pairIdx];
            if (count > 0) {
                std::set<IntersectionPoint> pair_intersections;

                for (unsigned int i = 0; i < count; ++i) {
                    unsigned int dataIdx = pairIdx * MAX_INTERSECTIONS_PER_PAIR + i;
                    const auto& pt = h_intersection_points[dataIdx];

                    pair_intersections.emplace(
                        pt.x,
                        pt.y,
                        pt.distanceThreshold,
                        PointType::REAL_INTERSECTION);
                }

                intersection_points_set[std::make_pair(subject_id, clipper_id)] = std::move(pair_intersections);
            }
        }
    }

    /*
    // OLD CPU CODE - Commented out for reference
    // Unpack pairs and insert into set
    for (unsigned int packed_pair : h_packed_pairs) {
        unsigned int subject_id = packed_pair & 0xFFFF;
        unsigned int clipper_id = (packed_pair >> 16) & 0xFFFF;
        intersecting_pairs.insert(std::make_pair(subject_id, clipper_id));
    }

    // Step 2: Compute intersection points for each intersecting polygon pair
    // This implements the real edge-edge intersection algorithm with angle-weighted thresholds
    for (const auto& pair : intersecting_pairs) {
        unsigned int subject_id = pair.first - 1;  // Convert from 1-based to 0-based
        unsigned int clipper_id = pair.second - 1;

        if (subject_id >= subjectLayer->polygonCount || clipper_id >= clipperLayer->polygonCount) {
            continue;
        }

        std::set<IntersectionPoint> pair_intersections;

        // Get polygon vertex data
        unsigned int subject_start = subjectLayer->h_startIndices[subject_id];
        unsigned int subject_count = subjectLayer->h_ptCounts[subject_id];
        unsigned int clipper_start = clipperLayer->h_startIndices[clipper_id];
        unsigned int clipper_count = clipperLayer->h_ptCounts[clipper_id];

        // Compute real edge-edge intersections with angle-weighted thresholds
        for (unsigned int si = 0; si < subject_count; ++si) {
            unsigned int next_si = (si + 1) % subject_count;

            cv::Point2f s1(subjectLayer->h_vertices[subject_start + si].x,
                           subjectLayer->h_vertices[subject_start + si].y);
            cv::Point2f s2(subjectLayer->h_vertices[subject_start + next_si].x,
                           subjectLayer->h_vertices[subject_start + next_si].y);

            for (unsigned int ci = 0; ci < clipper_count; ++ci) {
                unsigned int next_ci = (ci + 1) % clipper_count;

                cv::Point2f c1(clipperLayer->h_vertices[clipper_start + ci].x,
                               clipperLayer->h_vertices[clipper_start + ci].y);
                cv::Point2f c2(clipperLayer->h_vertices[clipper_start + next_ci].x,
                               clipperLayer->h_vertices[clipper_start + next_ci].y);

                // Compute line intersection
                cv::Point2f intersection;
                double angle_degrees = 90.0; // Default to perpendicular

                if (computeLineIntersection(s1, s2, c1, c2, intersection, angle_degrees)) {
                    // Calculate threshold based on angle (matches original algorithm)
                    float threshold = calculateDistanceThreshold(angle_degrees);

                    // Add intersection point with calculated threshold
                    pair_intersections.emplace(
                        static_cast<unsigned int>(std::round(intersection.x)),
                        static_cast<unsigned int>(std::round(intersection.y)),
                        threshold,
                        PointType::REAL_INTERSECTION
                    );
                }
            }
        }

        // Only store pairs that have real intersections
        if (!pair_intersections.empty()) {
            intersection_points_set[pair] = std::move(pair_intersections);
        }
    }
    */

    return intersection_points_set;
}

// ============================================================================
// GPU Kernel for Computing Intersection Points
// ============================================================================

// Each block handles one intersecting pair
// Assumes: 1) polygons have <= 1024 vertices, 2) max 32 intersection points per pair
// Device function to check if point p is on segment (a, b)
__device__ inline bool pointOnSegment_device(float px, float py, float ax, float ay, float bx, float by) {
    const float TOLERANCE = 1e-6f;

    // Check if point is within bounding box
    float minX = fminf(ax, bx) - TOLERANCE;
    float maxX = fmaxf(ax, bx) + TOLERANCE;
    float minY = fminf(ay, by) - TOLERANCE;
    float maxY = fmaxf(ay, by) + TOLERANCE;

    if (px < minX || px > maxX || py < minY || py > maxY) {
        return false;
    }

    // Check collinearity using cross product
    float dx1 = bx - ax;
    float dy1 = by - ay;
    float dx2 = px - ax;
    float dy2 = py - ay;

    float cross = dx1 * dy2 - dy1 * dx2;

    return fabsf(cross) < TOLERANCE;
}

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
    unsigned int* outputCounts) {

    const unsigned int MAX_INTERSECTIONS = 64;  // Increased to handle endpoint cases
    const unsigned int pairIdx = blockIdx.x;

    if (pairIdx >= numPairs) return;

    // Unpack the pair
    unsigned int packed = packedPairs[pairIdx];
    unsigned int subject_id = (packed & 0xFFFF) - 1;  // Convert to 0-based
    unsigned int clipper_id = ((packed >> 16) & 0xFFFF) - 1;

    // Load polygon metadata
    unsigned int subject_start = subjectStartIndices[subject_id];
    unsigned int subject_count = subjectPtCounts[subject_id];
    unsigned int clipper_start = clipperStartIndices[clipper_id];
    unsigned int clipper_count = clipperPtCounts[clipper_id];

    if (subject_count == 0 || clipper_count == 0 || subject_count > 1024 || clipper_count > 1024) {
        if (threadIdx.x == 0) {
            outputCounts[pairIdx] = 0;
        }
        return;
    }

    // Shared memory for vertices (assuming max 1024 vertices per polygon)
    __shared__ uint2 s_subject_verts[1024];
    __shared__ uint2 s_clipper_verts[1024];
    __shared__ unsigned int s_intersection_count;

    // Initialize shared intersection count
    if (threadIdx.x == 0) {
        s_intersection_count = 0;
    }
    __syncthreads();

    // Load subject vertices into shared memory
    for (unsigned int i = threadIdx.x; i < subject_count; i += blockDim.x) {
        s_subject_verts[i] = subjectVertices[subject_start + i];
    }

    // Load clipper vertices into shared memory
    for (unsigned int i = threadIdx.x; i < clipper_count; i += blockDim.x) {
        s_clipper_verts[i] = clipperVertices[clipper_start + i];
    }

    __syncthreads();

    // Each thread computes intersections for a subset of edge pairs
    // Total edge pairs = subject_count * clipper_count
    unsigned int totalEdgePairs = subject_count * clipper_count;

    for (unsigned int edgePairIdx = threadIdx.x; edgePairIdx < totalEdgePairs; edgePairIdx += blockDim.x) {
        unsigned int si = edgePairIdx / clipper_count;
        unsigned int ci = edgePairIdx % clipper_count;

        unsigned int next_si = (si + 1) % subject_count;
        unsigned int next_ci = (ci + 1) % clipper_count;

        // Get edge vertices
        float s1x = static_cast<float>(s_subject_verts[si].x);
        float s1y = static_cast<float>(s_subject_verts[si].y);
        float s2x = static_cast<float>(s_subject_verts[next_si].x);
        float s2y = static_cast<float>(s_subject_verts[next_si].y);

        float c1x = static_cast<float>(s_clipper_verts[ci].x);
        float c1y = static_cast<float>(s_clipper_verts[ci].y);
        float c2x = static_cast<float>(s_clipper_verts[next_ci].x);
        float c2y = static_cast<float>(s_clipper_verts[next_ci].y);

        // Compute edge-edge intersection
        float intersectX, intersectY, angle_degrees = 90.0f;

        if (computeLineIntersection_device(s1x, s1y, s2x, s2y, c1x, c1y, c2x, c2y,
                                           intersectX, intersectY, angle_degrees)) {
            // Calculate distance threshold based on angle
            float threshold = calculateDistanceThreshold_device(angle_degrees);

            // Atomically get slot for this intersection
            unsigned int slot = atomicAdd(&s_intersection_count, 1);

            if (slot < MAX_INTERSECTIONS) {
                // Store in global output
                unsigned int outputIdx = pairIdx * MAX_INTERSECTIONS + slot;
                outputPoints[outputIdx].x = static_cast<unsigned int>(roundf(intersectX));
                outputPoints[outputIdx].y = static_cast<unsigned int>(roundf(intersectY));
                outputPoints[outputIdx].distanceThreshold = threshold;
            }
        }

        // Check endpoint-on-segment cases (use 90° default angle for these)
        const float default_threshold = 2.0f;  // For perpendicular intersections

        // Check if clipper endpoints lie on subject edge
        if (pointOnSegment_device(c1x, c1y, s1x, s1y, s2x, s2y)) {
            unsigned int slot = atomicAdd(&s_intersection_count, 1);
            if (slot < MAX_INTERSECTIONS) {
                unsigned int outputIdx = pairIdx * MAX_INTERSECTIONS + slot;
                outputPoints[outputIdx].x = static_cast<unsigned int>(roundf(c1x));
                outputPoints[outputIdx].y = static_cast<unsigned int>(roundf(c1y));
                outputPoints[outputIdx].distanceThreshold = default_threshold;
            }
        }

        if (pointOnSegment_device(c2x, c2y, s1x, s1y, s2x, s2y)) {
            unsigned int slot = atomicAdd(&s_intersection_count, 1);
            if (slot < MAX_INTERSECTIONS) {
                unsigned int outputIdx = pairIdx * MAX_INTERSECTIONS + slot;
                outputPoints[outputIdx].x = static_cast<unsigned int>(roundf(c2x));
                outputPoints[outputIdx].y = static_cast<unsigned int>(roundf(c2y));
                outputPoints[outputIdx].distanceThreshold = default_threshold;
            }
        }

        // Check if subject endpoints lie on clipper edge
        if (pointOnSegment_device(s1x, s1y, c1x, c1y, c2x, c2y)) {
            unsigned int slot = atomicAdd(&s_intersection_count, 1);
            if (slot < MAX_INTERSECTIONS) {
                unsigned int outputIdx = pairIdx * MAX_INTERSECTIONS + slot;
                outputPoints[outputIdx].x = static_cast<unsigned int>(roundf(s1x));
                outputPoints[outputIdx].y = static_cast<unsigned int>(roundf(s1y));
                outputPoints[outputIdx].distanceThreshold = default_threshold;
            }
        }

        if (pointOnSegment_device(s2x, s2y, c1x, c1y, c2x, c2y)) {
            unsigned int slot = atomicAdd(&s_intersection_count, 1);
            if (slot < MAX_INTERSECTIONS) {
                unsigned int outputIdx = pairIdx * MAX_INTERSECTIONS + slot;
                outputPoints[outputIdx].x = static_cast<unsigned int>(roundf(s2x));
                outputPoints[outputIdx].y = static_cast<unsigned int>(roundf(s2y));
                outputPoints[outputIdx].distanceThreshold = default_threshold;
            }
        }
    }

    __syncthreads();

    // Write final count
    if (threadIdx.x == 0) {
        outputCounts[pairIdx] = min(s_intersection_count, MAX_INTERSECTIONS);
    }
}

} // namespace GpuLithoLib
