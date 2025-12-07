#pragma once

#include "../include/gpuLitho.h"
#include "../include/operation_types.h"
#include "IntersectionCompute.cuh"
#include <vector>
#include <map>
#include <set>
#include <opencv2/opencv.hpp>

namespace GpuLithoLib {

// Forward declarations
class LayerImpl;

using gpuLitho::OperationType;

// Structure for candidate points in contour simplification
struct CandidatePoint {
    unsigned int x;
    unsigned int y;
    float max_distance_threshold;

    CandidatePoint(unsigned int px, unsigned int py, float threshold)
        : x(px), y(py), max_distance_threshold(threshold) {}
};

/**
 * @brief Engine for simplifying raw contours using geometry information
 *
 * This engine handles:
 * - Simplifying raw contours using layer vertex information
 * - Matching contour points to intersection points and polygon vertices
 * - Creating visualization of the simplification process
 */
class SimplifyContourEngine {
public:
    SimplifyContourEngine();
    ~SimplifyContourEngine();

    /**
     * @brief Simplify contours using layer vertex information and intersection points
     * @param raw_contours Raw contours from RawContourDetectEngine
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
};

} // namespace GpuLithoLib
