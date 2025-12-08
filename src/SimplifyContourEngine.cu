#include "SimplifyContourEngine.cuh"
#include "LayerImpl.h"
#include "VisualizationUtils.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace GpuLithoLib {

// ============================================================================
// SimplifyContourEngine Implementation
// ============================================================================

SimplifyContourEngine::SimplifyContourEngine() {}

SimplifyContourEngine::~SimplifyContourEngine() {}

std::vector<std::vector<cv::Point>> SimplifyContourEngine::simplifyContoursWithGeometry(
    const std::vector<std::vector<cv::Point>>& raw_contours,
    LayerImpl* subjectLayer,
    LayerImpl* clipperLayer,
    LayerImpl* outputLayer,
    const std::map<std::pair<unsigned int, unsigned int>, std::set<IntersectionPoint>>& intersection_points_set,
    OperationType opType,
    unsigned int currentGridWidth,
    unsigned int currentGridHeight) {

    std::vector<std::vector<cv::Point>> simplified_contours;

    if (!outputLayer || !subjectLayer || !clipperLayer) {
        return simplified_contours;
    }

    // Ensure bitmap is on host
    outputLayer->copyBitmapToHost();

    // Process each contour using the exact algorithm from main.cu (lines 1283-1624)
    for (size_t contour_idx = 0; contour_idx < raw_contours.size(); ++contour_idx) {
        const auto& contour = raw_contours[contour_idx];
        if (contour.size() < 5) {
            continue;
        }

        // Determine which polygons this contour belongs to
        std::set<unsigned int> subject_ids;
        std::set<unsigned int> clipper_ids;

        // Sample points from the contour to determine associated polygon IDs
        for (size_t pt_idx = 0; pt_idx < contour.size(); ++pt_idx) {
            const cv::Point& pt = contour[pt_idx];

            // Check bounds
            if (pt.x >= 0 && pt.x < static_cast<int>(currentGridWidth) &&
                pt.y >= 0 && pt.y < static_cast<int>(currentGridHeight)) {
                unsigned int pixel_idx = pt.y * currentGridWidth + pt.x;
                unsigned int pixel_value = outputLayer->h_bitmap[pixel_idx];

                if (pixel_value > 0) {
                    // Extract subject and clipper IDs from pixel value
                    unsigned int clipper_id = (pixel_value >> 16) & 0xFFFF; // Upper 16 bits
                    unsigned int subject_id = pixel_value & 0xFFFF;         // Lower 16 bits

                    if (subject_id > 0) {
                        subject_ids.insert(subject_id);
                    }
                    if (clipper_id > 0) {
                        clipper_ids.insert(clipper_id);
                    }
                }

                // For DIFFERENCE operation, also check 4-neighbors to capture clipper polygon info
                if (opType == OperationType::DIFFERENCE) {
                    int dx[] = {-1, 1, 0, 0};
                    int dy[] = {0, 0, -1, 1};

                    for (int dir = 0; dir < 4; ++dir) {
                        int nx = pt.x + dx[dir];
                        int ny = pt.y + dy[dir];

                        if (nx >= 0 && nx < static_cast<int>(currentGridWidth) &&
                            ny >= 0 && ny < static_cast<int>(currentGridHeight)) {
                            unsigned int neighbor_idx = ny * currentGridWidth + nx;
                            unsigned int neighbor_value = outputLayer->h_bitmap[neighbor_idx];

                            if (neighbor_value > 0) {
                                unsigned int neighbor_clipper_id = (neighbor_value >> 16) & 0xFFFF;
                                unsigned int neighbor_subject_id = neighbor_value & 0xFFFF;

                                if (neighbor_subject_id > 0) {
                                    subject_ids.insert(neighbor_subject_id);
                                }
                                if (neighbor_clipper_id > 0) {
                                    clipper_ids.insert(neighbor_clipper_id);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Create a map from a contour point index to an intersection point or subject/clipper vertex
        std::map<size_t, std::pair<unsigned int, unsigned int>> contour_to_candidate_map;

        // Handle intersection case: both subject and clipper IDs present
        if (!subject_ids.empty() && !clipper_ids.empty()) {
            // Create comprehensive point vector including:
            // 1. All real intersection points from all possible subject-clipper pairs (with calculated thresholds)
            // 2. All subject vertices from subject_ids (threshold = 1.5f)
            // 3. All clipper vertices from clipper_ids (threshold = 1.5f)
            std::vector<CandidatePoint> all_candidate_points;

            // Collect all real intersection points from all possible subject-clipper pairs
            for (unsigned int subject_id : subject_ids) {
                for (unsigned int clipper_id : clipper_ids) {
                    auto pair_key = std::make_pair(subject_id, clipper_id);
                    auto pair_it = intersection_points_set.find(pair_key);

                    if (pair_it != intersection_points_set.end()) {
                        const auto &intersection_points = pair_it->second;

                        // Add all real intersection points with their calculated thresholds
                        for (const auto &intersection_point : intersection_points) {
                            all_candidate_points.emplace_back(
                                intersection_point.x,
                                intersection_point.y,
                                intersection_point.max_distance_threshold
                            );
                        }
                    }
                }
            }

            int real_intersection_count = all_candidate_points.size();

            // Collect all subject vertices with threshold 1.5f
            for (unsigned int subject_id : subject_ids) {
                if (subject_id > 0 && subject_id <= subjectLayer->polygonCount) {
                    unsigned int start_idx = subjectLayer->h_startIndices[subject_id - 1];
                    unsigned int vertex_count = subjectLayer->h_ptCounts[subject_id - 1];

                    for (unsigned int v = 0; v < vertex_count; ++v) {
                        unsigned int x = subjectLayer->h_vertices[start_idx + v].x;
                        unsigned int y = subjectLayer->h_vertices[start_idx + v].y;
                        all_candidate_points.emplace_back(x, y, 1.5f);
                    }
                }
            }

            // Collect all clipper vertices with threshold 1.5f
            for (unsigned int clipper_id : clipper_ids) {
                if (clipper_id > 0 && clipper_id <= clipperLayer->polygonCount) {
                    unsigned int start_idx = clipperLayer->h_startIndices[clipper_id - 1];
                    unsigned int vertex_count = clipperLayer->h_ptCounts[clipper_id - 1];

                    for (unsigned int v = 0; v < vertex_count; ++v) {
                        unsigned int x = clipperLayer->h_vertices[start_idx + v].x;
                        unsigned int y = clipperLayer->h_vertices[start_idx + v].y;
                        all_candidate_points.emplace_back(x, y, 1.5f);
                    }
                }
            }

            // Increase all max distance thresholds by 0.6f for XOR operation
            if (opType == OperationType::XOR) {
                for (auto &candidate_pt : all_candidate_points) {
                    candidate_pt.max_distance_threshold += 0.6f;
                }
            }

            // Create a vector to track which candidate points are matched
            std::vector<bool> candidate_point_matched(all_candidate_points.size(), false);

            // Unified matching stage: iterate through increasing distance thresholds
            for (size_t pt_idx = 0; pt_idx < all_candidate_points.size(); ++pt_idx) {
                if (candidate_point_matched[pt_idx])
                    continue;

                const auto &candidate_pt = all_candidate_points[pt_idx];
                float max_distance_threshold = candidate_pt.max_distance_threshold;

                // Iterate through threshold levels from 0 to max_distance_threshold
                for (float threshold = 0.0f; threshold <= max_distance_threshold; threshold += 0.6f) {
                    if (candidate_point_matched[pt_idx])
                        break;

                    for (size_t i = 0; i < contour.size(); ++i) {
                        // Skip if this contour point is already mapped
                        if (contour_to_candidate_map.find(i) != contour_to_candidate_map.end())
                            continue;

                        const cv::Point &contour_pt = contour[i];

                        // Calculate Euclidean distance
                        float dx = static_cast<float>(contour_pt.x) - static_cast<float>(candidate_pt.x);
                        float dy = static_cast<float>(contour_pt.y) - static_cast<float>(candidate_pt.y);
                        float distance = std::sqrt(dx * dx + dy * dy);

                        // Check if distance is within current threshold
                        if (distance <= threshold) {
                            contour_to_candidate_map[i] = std::make_pair(candidate_pt.x, candidate_pt.y);
                            candidate_point_matched[pt_idx] = true;
                            break;
                        }
                    }
                }
            }

            // Build simplified contour from mapped points
            std::vector<cv::Point> filtered_contour;
            for (size_t i = 0; i < contour.size(); ++i) {
                auto map_it = contour_to_candidate_map.find(i);
                if (map_it != contour_to_candidate_map.end()) {
                    const auto &intersection_pt = map_it->second;
                    filtered_contour.push_back(cv::Point(
                        static_cast<int>(intersection_pt.first),
                        static_cast<int>(intersection_pt.second)));
                }
            }

            if (filtered_contour.size() > 2) {
                simplified_contours.push_back(std::move(filtered_contour));
            }
            else //push raw contour if not enough vertices
            {
                simplified_contours.push_back(contour);
            }
        }
        // Handle pure subject or pure clipper cases
        else if (subject_ids.empty() && !clipper_ids.empty()) {
            // Pure clipper polygon - use clipper vertices directly
            unsigned int clipper_id = *clipper_ids.begin() - 1;
            if (clipper_id < clipperLayer->polygonCount) {
                unsigned int start = clipperLayer->h_startIndices[clipper_id];
                unsigned int count = clipperLayer->h_ptCounts[clipper_id];

                std::vector<cv::Point> vertices;
                for (unsigned int i = 0; i < count; ++i) {
                    vertices.emplace_back(
                        clipperLayer->h_vertices[start + i].x,
                        clipperLayer->h_vertices[start + i].y
                    );
                }
                if (vertices.size() > 2) {
                    simplified_contours.push_back(std::move(vertices));
                }
                else //push raw contour if not enough vertices
                {
                    simplified_contours.push_back(contour);
                }
            }
        }
        else if (!subject_ids.empty() && clipper_ids.empty()) {
            // Pure subject polygon - use subject vertices directly
            unsigned int subject_id = *subject_ids.begin() - 1;
            if (subject_id < subjectLayer->polygonCount) {
                unsigned int start = subjectLayer->h_startIndices[subject_id];
                unsigned int count = subjectLayer->h_ptCounts[subject_id];

                std::vector<cv::Point> vertices;
                for (unsigned int i = 0; i < count; ++i) {
                    vertices.emplace_back(
                        subjectLayer->h_vertices[start + i].x,
                        subjectLayer->h_vertices[start + i].y
                    );
                }
                if (vertices.size() > 2) {
                    simplified_contours.push_back(std::move(vertices));
                }
                else //push raw contour if not enough vertices
                {
                    simplified_contours.push_back(contour);
                }
            }
        }
    }

    return simplified_contours;
}

} // namespace GpuLithoLib
