#include "ContourProcessing.cuh"
#include "LayerImpl.h"
#include "GpuOperations.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace GpuLithoLib {

// ============================================================================
// ContourDetectEngine Implementation
// ============================================================================

ContourDetectEngine::ContourDetectEngine() {}

ContourDetectEngine::~ContourDetectEngine() {}

// ============================================================================
// Contour Detection Methods
// ============================================================================

// Detect raw contours from output layer bitmap
std::vector<std::vector<cv::Point>> ContourDetectEngine::detectRawContours(
    LayerImpl* outputLayer,
    OperationType opType,
    unsigned int currentGridWidth,
    unsigned int currentGridHeight) {
    std::vector<std::vector<cv::Point>> contours;

    if (!outputLayer) {
        return contours;
    }

    // NOTE: The GPU kernel approach for extracting contour pixels
    // This creates a bitmap with only contour pixels rendered

    // Ensure we have a contour bitmap
    auto contourLayer = std::make_unique<LayerImpl>();
    contourLayer->ensureBitmapAllocated(currentGridWidth, currentGridHeight);
    contourLayer->clearBitmap();

    // Extract contours using GPU kernel
    const int chunkDim = 30;
    dim3 blockSize(32, 32);
    dim3 gridSize(iDivUp(currentGridWidth, chunkDim), iDivUp(currentGridHeight, chunkDim));

    extractContours_kernel<<<gridSize, blockSize>>>(
        outputLayer->d_bitmap,
        contourLayer->d_bitmap,
        currentGridWidth,
        currentGridHeight,
        chunkDim,
        opType);

    CHECK_GPU_ERROR(gpuGetLastError());
    CHECK_GPU_ERROR(gpuDeviceSynchronize());

    // Copy to host
    contourLayer->copyBitmapToHost();

    // ========================================================================
    // GPU-based contour pixel sorting for parallel contour tracing
    // ========================================================================
    // Sort contour pixels by value to group same-valued pixels together
    // This prepares the data for parallel contour tracing where each thread
    // can process one group independently
    SortedContourPixels sortedPixels = sortContourPixelsByValue(
        contourLayer->d_bitmap,
        currentGridWidth,
        currentGridHeight);

    // Print statistics about sorted groups
    if (sortedPixels.count > 0) {
        std::cout << "Contour pixel sorting complete:" << std::endl;
        std::cout << "  Total non-zero pixels: " << sortedPixels.count << std::endl;

        // Count unique values (groups) by checking value changes
        thrust::device_vector<unsigned int> uniqueValues(sortedPixels.count);
        auto uniqueEnd = thrust::unique_copy(
            sortedPixels.values.begin(),
            sortedPixels.values.end(),
            uniqueValues.begin());
        unsigned int numGroups = thrust::distance(uniqueValues.begin(), uniqueEnd);
        std::cout << "  Number of unique pixel value groups: " << numGroups << std::endl;
    }

    // ========================================================================
    // GPU Contour Tracing for INTERSECTION operation
    // ========================================================================
    std::vector<std::vector<cv::Point>> gpuContours;
    if (opType == OperationType::INTERSECTION && sortedPixels.count > 0) {
        std::cout << "Using GPU contour tracing for INTERSECTION operation" << std::endl;

        // Trace contours using GPU parallel algorithm
        gpuContours = traceContoursGPU(
            sortedPixels,
            contourLayer->d_bitmap,
            currentGridWidth,
            currentGridHeight);

        std::cout << "GPU tracing found " << gpuContours.size() << " contours" << std::endl;

        // Continue to also run OpenCV for comparison
        // Don't return early - we'll compare both methods
    }

    // Ensure output bitmap is on host
    outputLayer->copyBitmapToHost();

    // Create binary image directly from outputLayer bitmap based on operation type
    // This matches the correct implementation from the old code
    cv::Mat binaryImage(currentGridHeight, currentGridWidth, CV_8UC1);
    for (unsigned int y = 0; y < currentGridHeight; ++y) {
        for (unsigned int x = 0; x < currentGridWidth; ++x) {
            unsigned int idx = y * currentGridWidth + x;
            unsigned int val = outputLayer->h_bitmap[idx];

            switch(opType) {
                case OperationType::OFFSET:
                {
                    binaryImage.at<uchar>(y, x) = (val & 0xFFFF) ? 255 : 0;
                    break;
                }
                case OperationType::INTERSECTION:
                {
                    binaryImage.at<uchar>(y, x) = ((val & 0xFFFF) && (val & 0xFFFF0000)) ? 255 : 0;
                    break;
                }
                case OperationType::UNION:
                {
                    binaryImage.at<uchar>(y, x) = ((val & 0xFFFF) || (val & 0xFFFF0000)) ? 255 : 0;
                    break;
                }
                case OperationType::DIFFERENCE:
                {
                    binaryImage.at<uchar>(y, x) = ((val & 0xFFFF) && !(val & 0xFFFF0000)) ? 255 : 0;
                    break;
                }
                default:
                {
                    binaryImage.at<uchar>(y, x) = (val > 0) ? 255 : 0;
                    break;
                }
            }
        }
    }

    // Save binary image for debugging
    std::string debugFilename = "debug_binary_";
    switch(opType) {
        case OperationType::OFFSET:
            debugFilename += "offset.png";
            break;
        case OperationType::INTERSECTION:
            debugFilename += "intersection.png";
            break;
        case OperationType::UNION:
            debugFilename += "union.png";
            break;
        case OperationType::DIFFERENCE:
            debugFilename += "difference.png";
            break;
        case OperationType::XOR:
            debugFilename += "xor.png";
            break;
        default:
            debugFilename += "unknown.png";
            break;
    }
    cv::imwrite(debugFilename, binaryImage);

    // Use OpenCV to find contours
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);

    // ========================================================================
    // Comparison Visualization for INTERSECTION operation
    // ========================================================================
    if (opType == OperationType::INTERSECTION && !gpuContours.empty()) {
        std::cout << "Creating comparison visualization: GPU vs OpenCV contours" << std::endl;
        std::cout << "  GPU contours: " << gpuContours.size() << std::endl;
        std::cout << "  OpenCV contours: " << contours.size() << std::endl;

        // Create RGB image for comparison
        cv::Mat comparisonImage = cv::Mat::zeros(currentGridHeight, currentGridWidth, CV_8UC3);

        // Draw OpenCV contours in GREEN using polylines
        cv::Scalar greenColor(0, 255, 0); // GREEN (BGR format)
        for (size_t i = 0; i < contours.size(); ++i) {
            if (contours[i].size() > 1) {
                cv::polylines(comparisonImage, contours[i], true, greenColor, 1, cv::LINE_8);
            }
        }

        // Draw GPU contours in RED using polylines (overlay on top)
        cv::Scalar redColor(0, 0, 255); // RED (BGR format)
        for (size_t i = 0; i < gpuContours.size(); ++i) {
            if (gpuContours[i].size() > 1) {
                cv::polylines(comparisonImage, gpuContours[i], true, redColor, 1, cv::LINE_8);
            }
        }

        // Save comparison image
        std::string comparisonFilename = "contour_comparison_intersection.png";
        cv::imwrite(comparisonFilename, comparisonImage);
        std::cout << "  Saved comparison to: " << comparisonFilename << std::endl;
        std::cout << "  Color legend: GREEN=OpenCV contours, RED=GPU contours (overlay)" << std::endl;
        std::cout << "  Note: If GREEN is visible, GPU is missing that part" << std::endl;

        // Return GPU contours for INTERSECTION since that's the new implementation
        return gpuContours;
    }

    return contours;
}

std::vector<std::vector<cv::Point>> ContourDetectEngine::simplifyContoursWithGeometry(
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
            if (pt.x >= 0 && pt.x < currentGridWidth && pt.y >= 0 && pt.y < currentGridHeight) {
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
                // because clipper edges might not be captured at the exact contour boundary
                if (opType == OperationType::DIFFERENCE) {
                    // Define 4-neighbor offsets: left, right, up, down
                    int dx[] = {-1, 1, 0, 0};
                    int dy[] = {0, 0, -1, 1};

                    for (int dir = 0; dir < 4; ++dir) {
                        int nx = pt.x + dx[dir];
                        int ny = pt.y + dy[dir];

                        // Check if neighbor is within bounds
                        if (nx >= 0 && nx < currentGridWidth && ny >= 0 && ny < currentGridHeight) {
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
        // In the end of simplification, we will only keep contour points that are in this map
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
                    unsigned int start_idx = subjectLayer->h_startIndices[subject_id - 1]; // Convert to 0-based
                    unsigned int vertex_count = subjectLayer->h_ptCounts[subject_id - 1];

                    for (unsigned int v = 0; v < vertex_count; ++v) {
                        unsigned int x = subjectLayer->h_vertices[start_idx + v].x;
                        unsigned int y = subjectLayer->h_vertices[start_idx + v].y;
                        all_candidate_points.emplace_back(x, y, 1.5f); // Subject vertex threshold
                    }
                }
            }

            // Collect all clipper vertices with threshold 1.5f
            for (unsigned int clipper_id : clipper_ids) {
                if (clipper_id > 0 && clipper_id <= clipperLayer->polygonCount) {
                    unsigned int start_idx = clipperLayer->h_startIndices[clipper_id - 1]; // Convert to 0-based
                    unsigned int vertex_count = clipperLayer->h_ptCounts[clipper_id - 1];

                    for (unsigned int v = 0; v < vertex_count; ++v) {
                        unsigned int x = clipperLayer->h_vertices[start_idx + v].x;
                        unsigned int y = clipperLayer->h_vertices[start_idx + v].y;
                        all_candidate_points.emplace_back(x, y, 1.5f); // Clipper vertex threshold
                    }
                }
            }

            // Increase all max distance thresholds by 0.6f to allow more flexible matching for XOR operation
            if (opType == OperationType::XOR) {
                for (auto &candidate_pt : all_candidate_points) {
                    candidate_pt.max_distance_threshold += 0.6f;
                }
            }

            // Create a vector to track which candidate points are matched to contour points
            std::vector<bool> candidate_point_matched(all_candidate_points.size(), false);

            // Unified matching stage: iterate through increasing distance thresholds
            // For each candidate point, try to match it with contour points using progressively larger thresholds
            for (size_t pt_idx = 0; pt_idx < all_candidate_points.size(); ++pt_idx) {
                if (candidate_point_matched[pt_idx])
                    continue;

                const auto &candidate_pt = all_candidate_points[pt_idx];
                float max_distance_threshold = candidate_pt.max_distance_threshold;

                // Iterate through threshold levels from 0 to max_distance_threshold
                // Use step size of 0.6f to cover: 0, 0.6, 1.2, 1.8, 2.4, ... up to max
                for (float threshold = 0.0f; threshold <= max_distance_threshold; threshold += 0.6f) {
                    if (candidate_point_matched[pt_idx])
                        break; // Already matched at a smaller threshold
                    
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
                    // Use the intersection point coordinates instead of contour point
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
        // Handle pure subject or pure clipper cases (unchanged from original)
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

// ============================================================================
// GPU Contour Pixel Sorting Implementation
// ============================================================================

SortedContourPixels ContourDetectEngine::sortContourPixelsByValue(
    const unsigned int* contourBitmap,
    unsigned int width,
    unsigned int height) {

    SortedContourPixels result;

    unsigned int totalPixels = width * height;

    // Step 1: Create index array [0, 1, 2, ..., totalPixels-1]
    thrust::device_vector<unsigned int> d_indices(totalPixels);
    thrust::sequence(d_indices.begin(), d_indices.end(), 0);

    // Step 2: Copy bitmap to device_vector for processing
    thrust::device_vector<unsigned int> d_values(totalPixels);
    thrust::copy(
        thrust::device_pointer_cast(contourBitmap),
        thrust::device_pointer_cast(contourBitmap + totalPixels),
        d_values.begin());

    // Step 3: Remove elements where bitmap value is 0
    // We'll use remove_if with a custom predicate
    auto new_end = thrust::remove_if(
        thrust::make_zip_iterator(thrust::make_tuple(d_values.begin(), d_indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_values.end(), d_indices.end())),
        d_values.begin(),
        thrust::placeholders::_1 == 0);

    // Calculate the number of non-zero pixels
    unsigned int nonZeroCount = thrust::distance(
        thrust::make_zip_iterator(thrust::make_tuple(d_values.begin(), d_indices.begin())),
        new_end);

    // Resize vectors to actual size
    d_values.resize(nonZeroCount);
    d_indices.resize(nonZeroCount);

    // Step 4: Sort indices by bitmap values (using values as keys)
    thrust::sort_by_key(d_values.begin(), d_values.end(), d_indices.begin());

    // Step 5: Store results
    result.values = std::move(d_values);
    result.indices = std::move(d_indices);
    result.count = nonZeroCount;

    std::cout << "Sorted " << nonZeroCount << " non-zero contour pixels into groups by value" << std::endl;

    return result;
}

// ============================================================================
// GPU Contour Tracing Implementation
// ============================================================================

std::vector<std::vector<cv::Point>> ContourDetectEngine::traceContoursGPU(
    const SortedContourPixels& sortedPixels,
    const unsigned int* contourBitmap,
    unsigned int width,
    unsigned int height) {

    std::vector<std::vector<cv::Point>> contours;

    if (sortedPixels.count == 0) {
        return contours;
    }

    // Step 1: Identify groups (consecutive pixels with same value)
    std::vector<GroupInfo> h_groups;
    thrust::host_vector<unsigned int> h_values = sortedPixels.values;

    unsigned int currentValue = h_values[0];
    unsigned int groupStart = 0;

    for (unsigned int i = 1; i <= sortedPixels.count; ++i) {
        bool isNewGroup = (i == sortedPixels.count) || (h_values[i] != currentValue);

        if (isNewGroup) {
            GroupInfo group;
            group.value = currentValue;
            group.startIdx = groupStart;
            group.count = i - groupStart;
            h_groups.push_back(group);

            if (i < sortedPixels.count) {
                currentValue = h_values[i];
                groupStart = i;
            }
        }
    }

    unsigned int numGroups = h_groups.size();
    std::cout << "Found " << numGroups << " pixel value groups for tracing" << std::endl;

    if (numGroups == 0) {
        return contours;
    }

    // Step 2: Allocate GPU memory for groups
    thrust::device_vector<GroupInfo> d_groups(h_groups);

    // Step 3: Allocate visited array (one flag per sorted pixel)
    thrust::device_vector<unsigned char> d_visited(sortedPixels.count, 0);

    // Step 4: Allocate output arrays
    const unsigned int maxPointsPerContour = 1<<13; // 8192 points max per contour
    thrust::device_vector<ContourPoint> d_outputContours(numGroups * maxPointsPerContour);
    thrust::device_vector<unsigned int> d_outputCounts(numGroups, 0);

    // Step 5: Launch kernel (one block per group)
    dim3 blockSize(1, 1, 1); // Using single thread per block for now
    dim3 gridSize(numGroups, 1, 1);

    traceContoursParallel_kernel<<<gridSize, blockSize>>>(
        contourBitmap,
        thrust::raw_pointer_cast(sortedPixels.indices.data()),
        thrust::raw_pointer_cast(sortedPixels.values.data()),
        thrust::raw_pointer_cast(d_groups.data()),
        numGroups,
        thrust::raw_pointer_cast(d_visited.data()),
        thrust::raw_pointer_cast(d_outputContours.data()),
        thrust::raw_pointer_cast(d_outputCounts.data()),
        width,
        height,
        maxPointsPerContour);

    CHECK_GPU_ERROR(gpuGetLastError());
    CHECK_GPU_ERROR(gpuDeviceSynchronize());

    // Step 6: Copy results back to host
    thrust::host_vector<ContourPoint> h_outputContours = d_outputContours;
    thrust::host_vector<unsigned int> h_outputCounts = d_outputCounts;

    // Step 7: Convert to cv::Point format
    for (unsigned int g = 0; g < numGroups; ++g) {
        unsigned int count = h_outputCounts[g];
        if (count > 0) {
            std::vector<cv::Point> contour;
            contour.reserve(count);

            for (unsigned int i = 0; i < count; ++i) {
                ContourPoint pt = h_outputContours[g * maxPointsPerContour + i];
                contour.push_back(cv::Point(pt.x, pt.y));
            }

            contours.push_back(std::move(contour));
        }
    }

    std::cout << "GPU tracing complete: extracted " << contours.size() << " contours" << std::endl;

    return contours;
}

} // namespace GpuLithoLib
