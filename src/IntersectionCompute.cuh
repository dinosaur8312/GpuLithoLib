#pragma once

#include "../include/gpuLitho.h"
#include "CommonRenderUtils.cuh"
#include <set>
#include <map>
#include <opencv2/opencv.hpp>

namespace GpuLithoLib {

// Forward declarations
class LayerImpl;

// === Intersection Point Types ===
// These must be defined before IntersectionComputeEngine class

enum class PointType {
    SUBJECT_VERTEX,
    CLIPPER_VERTEX,
    REAL_INTERSECTION
};

struct IntersectionPoint {
    unsigned int x;
    unsigned int y;
    float max_distance_threshold;
    PointType type;

    IntersectionPoint(unsigned int x_, unsigned int y_, float threshold, PointType t = PointType::REAL_INTERSECTION)
        : x(x_), y(y_), max_distance_threshold(threshold), type(t) {}

    bool operator<(const IntersectionPoint& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return max_distance_threshold < other.max_distance_threshold;
    }
};

// Functor for filtering intersection pixels (both subject_id and clipper_id non-zero)
struct IsIntersectionPixel {
    __host__ __device__
    bool operator()(unsigned int pixel_value) const {
        unsigned int subject_id = pixel_value & 0xFFFF;
        unsigned int clipper_id = (pixel_value >> 16) & 0xFFFF;
        return (subject_id > 0) && (clipper_id > 0);
    }
};

// IntersectionComputeEngine: Manages GPU-accelerated intersection computation
// Stores persistent GPU data for use in contour processing
class IntersectionComputeEngine {
public:
    // GPU-side data (persistent across operations)
    unsigned int* d_packed_pairs;           // Device array of packed (clipper_id << 16 | subject_id)
    IntersectionPointData* d_intersection_points;  // Device array [numPairs * 32] intersection points
    unsigned int* d_intersection_counts;    // Device array [numPairs] count of intersections per pair
    unsigned int numIntersectingPairs;      // Number of intersecting pairs

    IntersectionComputeEngine();
    ~IntersectionComputeEngine();

    // Free all GPU memory
    void freeData();

    // Compute all intersection points from overlay bitmap
    std::map<std::pair<unsigned int, unsigned int>, std::set<IntersectionPoint>>
    computeAllIntersectionPoints(
        LayerImpl* outputLayer,
        LayerImpl* subjectLayer,
        LayerImpl* clipperLayer,
        unsigned int currentGridWidth,
        unsigned int currentGridHeight);
};

// === Helper Functions ===

/**
 * @brief Calculate distance threshold based on intersection angle
 * Uses logarithmic interpolation for acute angles
 */
float calculateDistanceThreshold(double angle_degrees);

/**
 * @brief Compute intersection between two line segments
 * @return true if intersection exists within both segments
 */
bool computeLineIntersection(
    const cv::Point2f& p1, const cv::Point2f& p2,
    const cv::Point2f& p3, const cv::Point2f& p4,
    cv::Point2f& intersection, double& angle_degrees);

// ============================================================================
// GPU Kernel Declaration
// ============================================================================

/**
 * @brief GPU kernel to compute intersection points for polygon pairs
 * Each block handles one intersecting pair
 * Assumes: 1) polygons have <= 1024 vertices, 2) max 32 intersection points per pair
 *
 * @param packedPairs Array of packed pair IDs (device)
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
