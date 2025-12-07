#include "SimplifyContourEngine.cuh"
#include "LayerImpl.h"
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

    // ========================================================================
    // Visualization: Subject, Clipper, Raw Contours, and Intersection Points
    // ========================================================================
    cv::Mat debugImage = cv::Mat::zeros(currentGridHeight, currentGridWidth, CV_8UC3);

    // Layer 1 (bottom): Draw subject layer polygons in BLUE (line width 1)
    cv::Scalar blueColor(255, 0, 0);  // BGR format
    for (unsigned int polyIdx = 0; polyIdx < subjectLayer->polygonCount; ++polyIdx) {
        unsigned int start = subjectLayer->h_startIndices[polyIdx];
        unsigned int count = subjectLayer->h_ptCounts[polyIdx];

        std::vector<cv::Point> polyPoints;
        for (unsigned int i = 0; i < count; ++i) {
            uint2 v = subjectLayer->h_vertices[start + i];
            polyPoints.push_back(cv::Point(v.x, v.y));
        }
        if (polyPoints.size() > 1) {
            cv::polylines(debugImage, polyPoints, true, blueColor, 1, cv::LINE_8);

            // Add polygon ID label in blue at first vertex
            std::string label = "S" + std::to_string(polyIdx);
            cv::Point labelPos(polyPoints[0].x + 3, polyPoints[0].y - 3);
            cv::putText(debugImage, label, labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.3,
                       blueColor, 1, cv::LINE_AA);
        }
    }

    // Layer 2: Draw clipper layer polygons in RED (line width 1)
    cv::Scalar redColor(0, 0, 255);  // BGR format
    for (unsigned int polyIdx = 0; polyIdx < clipperLayer->polygonCount; ++polyIdx) {
        unsigned int start = clipperLayer->h_startIndices[polyIdx];
        unsigned int count = clipperLayer->h_ptCounts[polyIdx];

        std::vector<cv::Point> polyPoints;
        for (unsigned int i = 0; i < count; ++i) {
            uint2 v = clipperLayer->h_vertices[start + i];
            polyPoints.push_back(cv::Point(v.x, v.y));
        }
        if (polyPoints.size() > 1) {
            cv::polylines(debugImage, polyPoints, true, redColor, 1, cv::LINE_8);

            // Add polygon ID label in red at first vertex
            std::string label = "C" + std::to_string(polyIdx);
            cv::Point labelPos(polyPoints[0].x + 3, polyPoints[0].y - 3);
            cv::putText(debugImage, label, labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.3,
                       redColor, 1, cv::LINE_AA);
        }
    }

    // Layer 3: Draw raw contour pixels in GREEN (pixel size 1)
    cv::Scalar greenColor(0, 255, 0);  // BGR format
    for (const auto& contour : raw_contours) {
        for (const auto& pt : contour) {
            if (pt.x >= 0 && pt.x < static_cast<int>(currentGridWidth) &&
                pt.y >= 0 && pt.y < static_cast<int>(currentGridHeight)) {
                debugImage.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(0, 255, 0);  // Green pixel
            }
        }
    }

    // Layer 4 (top): Draw intersection points in YELLOW (circle size 2)
    cv::Scalar yellowColor(0, 255, 255);  // BGR format
    for (const auto& pair_entry : intersection_points_set) {
        for (const auto& int_pt : pair_entry.second) {
            cv::Point center(int_pt.x, int_pt.y);
            if (center.x >= 0 && center.x < static_cast<int>(currentGridWidth) &&
                center.y >= 0 && center.y < static_cast<int>(currentGridHeight)) {
                cv::circle(debugImage, center, 2, yellowColor, -1, cv::LINE_AA);
            }
        }
    }

    // Add grid and axis labels
    cv::Scalar gridColor(50, 50, 50);  // Dark gray grid
    cv::Scalar axisColor(150, 150, 150);  // Light gray for axis labels

    // Draw vertical grid lines every 200 pixels
    for (unsigned int x = 0; x <= currentGridWidth; x += 200) {
        cv::line(debugImage, cv::Point(x, 0), cv::Point(x, currentGridHeight - 1), gridColor, 1);

        // Add X-axis label
        if (x > 0) {
            std::string label = std::to_string(x);
            cv::putText(debugImage, label, cv::Point(x - 15, 15),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, axisColor, 1, cv::LINE_AA);
        }
    }

    // Draw horizontal grid lines every 200 pixels
    for (unsigned int y = 0; y <= currentGridHeight; y += 200) {
        cv::line(debugImage, cv::Point(0, y), cv::Point(currentGridWidth - 1, y), gridColor, 1);

        // Add Y-axis label
        if (y > 0) {
            std::string label = std::to_string(y);
            cv::putText(debugImage, label, cv::Point(5, y + 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, axisColor, 1, cv::LINE_AA);
        }
    }

    // Save the debug visualization
    cv::imwrite("simplification_debug_layers.png", debugImage);
    std::cout << "Saved simplification debug visualization: simplification_debug_layers.png" << std::endl;
    std::cout << "  Layers (bottom to top): BLUE=Subject, RED=Clipper, GREEN=Raw contours, YELLOW=Intersection points" << std::endl;
    std::cout << "  Grid: 200x200 pixels with axis labels" << std::endl;
    // ========================================================================

    // Process each contour using the exact algorithm from main.cu (lines 1283-1624)
    for (size_t contour_idx = 0; contour_idx < raw_contours.size(); ++contour_idx) {
        const auto& contour = raw_contours[contour_idx];
        if (contour.size() < 3) {
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
            }
        }
    }

    return simplified_contours;
}

} // namespace GpuLithoLib
