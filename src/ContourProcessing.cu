#include "ContourProcessing.cuh"
#include "GpuKernelProfiler.cuh"
#include "LayerImpl.h"
#include "GpuOperations.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace GpuLithoLib {

// External profiler instance (owned by GpuLithoEngine)
extern GpuKernelProfiler* g_kernelProfiler;

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

    // Time the extractContours kernel
    gpuEvent_t extractStart, extractStop;
    gpuEventCreate(&extractStart);
    gpuEventCreate(&extractStop);
    gpuEventRecord(extractStart);

    extractContours_kernel<<<gridSize, blockSize>>>(
        outputLayer->d_bitmap,
        contourLayer->d_bitmap,
        currentGridWidth,
        currentGridHeight,
        chunkDim,
        opType);

    CHECK_GPU_ERROR(gpuGetLastError());
    gpuEventRecord(extractStop);
    gpuEventSynchronize(extractStop);

    float extractMs = 0.0f;
    gpuEventElapsedTime(&extractMs, extractStart, extractStop);
    if (g_kernelProfiler) g_kernelProfiler->addExtractContoursTime(extractMs);

    gpuEventDestroy(extractStart);
    gpuEventDestroy(extractStop);

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
            outputLayer->d_bitmap,  // Pass overlay bitmap for polygon ID collection
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

        // Save comparison image (CPU first, GPU on top)
        std::string comparisonFilename = "contour_comparison_intersection_cpu_gpu.png";
        cv::imwrite(comparisonFilename, comparisonImage);
        std::cout << "  Saved comparison to: " << comparisonFilename << std::endl;
        std::cout << "  Color legend: GREEN=OpenCV contours (bottom), RED=GPU contours (overlay on top)" << std::endl;
        std::cout << "  Note: If GREEN is visible, GPU is missing that part" << std::endl;

        // Create reverse order comparison image (GPU first, CPU on top)
        cv::Mat reverseComparisonImage = cv::Mat::zeros(currentGridHeight, currentGridWidth, CV_8UC3);

        // Draw GPU contours in RED first (bottom layer)
        for (size_t i = 0; i < gpuContours.size(); ++i) {
            if (gpuContours[i].size() > 1) {
                cv::polylines(reverseComparisonImage, gpuContours[i], true, redColor, 1, cv::LINE_8);
            }
        }

        // Draw OpenCV contours in GREEN on top (overlay)
        for (size_t i = 0; i < contours.size(); ++i) {
            if (contours[i].size() > 1) {
                cv::polylines(reverseComparisonImage, contours[i], true, greenColor, 1, cv::LINE_8);
            }
        }

        // Save reverse order comparison image
        std::string reverseComparisonFilename = "contour_comparison_intersection_gpu_cpu.png";
        cv::imwrite(reverseComparisonFilename, reverseComparisonImage);
        std::cout << "  Saved reverse comparison to: " << reverseComparisonFilename << std::endl;
        std::cout << "  Color legend: RED=GPU contours (bottom), GREEN=OpenCV contours (overlay on top)" << std::endl;
        std::cout << "  Note: If RED is visible, OpenCV is missing that part" << std::endl;

        // Return GPU contours for INTERSECTION since that's the new implementation
     //   return gpuContours;
        return contours;
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
    auto values_new_end = new_end.get_iterator_tuple().template get<0>();
    unsigned int nonZeroCount = thrust::distance(d_values.begin(), values_new_end);

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
    const unsigned int* overlayBitmap,
    unsigned int width,
    unsigned int height) {

    std::vector<std::vector<cv::Point>> contours;

    if (sortedPixels.count == 0) {
        return contours;
    }

    // Step 1: Identify groups (consecutive pixels with same value) using CUB Run-Length Encode
    // OPTIMIZATION: Use CUB DeviceRunLengthEncode instead of Thrust reduce_by_key
    // This is simpler, more efficient, and purpose-built for exactly this use case
    thrust::device_vector<unsigned int> d_unique_values(sortedPixels.count);
    thrust::device_vector<unsigned int> d_group_counts(sortedPixels.count);
    thrust::device_vector<unsigned int> d_num_runs(1);

    // Determine temporary storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage, temp_storage_bytes,
        thrust::raw_pointer_cast(sortedPixels.values.data()),
        thrust::raw_pointer_cast(d_unique_values.data()),
        thrust::raw_pointer_cast(d_group_counts.data()),
        thrust::raw_pointer_cast(d_num_runs.data()),
        sortedPixels.count);

    // Allocate temporary storage
    CHECK_GPU_ERROR(gpuMalloc(&d_temp_storage, temp_storage_bytes));

    // Run encoding: [1,1,1,2,2,3] -> unique=[1,2,3], counts=[3,2,1], num_runs=3
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage, temp_storage_bytes,
        thrust::raw_pointer_cast(sortedPixels.values.data()),
        thrust::raw_pointer_cast(d_unique_values.data()),
        thrust::raw_pointer_cast(d_group_counts.data()),
        thrust::raw_pointer_cast(d_num_runs.data()),
        sortedPixels.count);

    // Get number of groups (copy single int from device)
    unsigned int numGroups;
    CHECK_GPU_ERROR(gpuMemcpy(&numGroups, thrust::raw_pointer_cast(d_num_runs.data()),
                              sizeof(unsigned int), gpuMemcpyDeviceToHost));

    CHECK_GPU_ERROR(gpuFree(d_temp_storage));

    std::cout << "Found " << numGroups << " pixel value groups for tracing" << std::endl;

    if (numGroups == 0) {
        return contours;
    }

    // Resize to actual number of groups
    d_unique_values.resize(numGroups);
    d_group_counts.resize(numGroups);

    // Step 2: Compute start indices using exclusive scan of group counts
    // Using separate pointer representation instead of GroupInfo struct
    thrust::device_vector<unsigned int> d_group_starts(numGroups);
    thrust::exclusive_scan(d_group_counts.begin(), d_group_counts.end(), d_group_starts.begin());

    // Now we have three separate arrays for group data:
    // - d_unique_values: pixel values for each group
    // - d_group_starts: starting index in sorted arrays for each group
    // - d_group_counts: number of pixels in each group

    // Step 3: Allocate visited array (one flag per sorted pixel)
    thrust::device_vector<unsigned char> d_visited(sortedPixels.count, 0);

    // Step 4: Allocate output arrays
    const unsigned int maxPointsPerContour = 1<<13; // 8192 points max per contour
    const unsigned int maxIDsPerGroup = 256;
    thrust::device_vector<uint2> d_outputContours(numGroups * maxPointsPerContour);
    thrust::device_vector<unsigned int> d_outputCounts(numGroups, 0);

    // Allocate separate arrays for polygon IDs (instead of ContourPolygonIDs struct)
    thrust::device_vector<unsigned int> d_subject_ids(numGroups * maxIDsPerGroup);
    thrust::device_vector<unsigned int> d_clipper_ids(numGroups * maxIDsPerGroup);
    thrust::device_vector<unsigned int> d_subject_counts(numGroups, 0);
    thrust::device_vector<unsigned int> d_clipper_counts(numGroups, 0);

    // ========== DEBUG: Visualize Group 11 contour pixels ==========
    const unsigned int DEBUG_GROUP = 11;
    if (DEBUG_GROUP < numGroups) {
        // Copy data to host
        thrust::host_vector<unsigned int> h_group_starts = d_group_starts;
        thrust::host_vector<unsigned int> h_group_counts = d_group_counts;
        thrust::host_vector<unsigned int> h_unique_values = d_unique_values;
        thrust::host_vector<unsigned int> h_sortedIndices = sortedPixels.indices;

        unsigned int groupStart = h_group_starts[DEBUG_GROUP];
        unsigned int groupCount = h_group_counts[DEBUG_GROUP];
        unsigned int groupValue = h_unique_values[DEBUG_GROUP];

        std::cout << "DEBUG Group " << DEBUG_GROUP << ": value=" << groupValue
                  << ", start=" << groupStart << ", count=" << groupCount << std::endl;

        // Find bounding box of group 11 pixels
        unsigned int minX = width, maxX = 0, minY = height, maxY = 0;
        for (unsigned int i = 0; i < groupCount; ++i) {
            unsigned int pixelIdx = h_sortedIndices[groupStart + i];
            unsigned int px = pixelIdx % width;
            unsigned int py = pixelIdx / width;
            minX = std::min(minX, px);
            maxX = std::max(maxX, px);
            minY = std::min(minY, py);
            maxY = std::max(maxY, py);
        }

        // Add margin for better visualization
        unsigned int margin = 5;
        minX = (minX > margin) ? (minX - margin) : 0;
        minY = (minY > margin) ? (minY - margin) : 0;
        maxX = std::min(maxX + margin, width - 1);
        maxY = std::min(maxY + margin, height - 1);

        unsigned int regionWidth = maxX - minX + 1;
        unsigned int regionHeight = maxY - minY + 1;

        std::cout << "  Bounding box: (" << minX << "," << minY << ") to ("
                  << maxX << "," << maxY << ")" << std::endl;

        // Create visualization image (scale up for better visibility)
        unsigned int scale = 10;
        cv::Mat visImage = cv::Mat::zeros(regionHeight * scale, regionWidth * scale, CV_8UC3);

        // Draw all pixels in group 11
        for (unsigned int i = 0; i < groupCount; ++i) {
            unsigned int pixelIdx = h_sortedIndices[groupStart + i];
            unsigned int px = pixelIdx % width;
            unsigned int py = pixelIdx / width;

            if (px >= minX && px <= maxX && py >= minY && py <= maxY) {
                unsigned int imgX = (px - minX) * scale;
                unsigned int imgY = (py - minY) * scale;
                cv::rectangle(visImage,
                            cv::Point(imgX, imgY),
                            cv::Point(imgX + scale - 1, imgY + scale - 1),
                            cv::Scalar(255, 255, 255), cv::FILLED);
            }
        }

        // Draw grid lines and axis labels
        for (unsigned int x = 0; x <= regionWidth; ++x) {
            int imgX = x * scale;
            cv::line(visImage, cv::Point(imgX, 0), cv::Point(imgX, regionHeight * scale - 1),
                    cv::Scalar(50, 50, 50), 1);

            // X-axis labels every 5 pixels
            if (x % 5 == 0 && x < regionWidth) {
                std::string label = std::to_string(minX + x);
                cv::putText(visImage, label, cv::Point(imgX + 2, 12),
                           cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
            }
        }
        for (unsigned int y = 0; y <= regionHeight; ++y) {
            int imgY = y * scale;
            cv::line(visImage, cv::Point(0, imgY), cv::Point(regionWidth * scale - 1, imgY),
                    cv::Scalar(50, 50, 50), 1);

            // Y-axis labels every 5 pixels
            if (y % 5 == 0 && y < regionHeight) {
                std::string label = std::to_string(minY + y);
                cv::putText(visImage, label, cv::Point(2, imgY + 12),
                           cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1);
            }
        }

        std::string filename = "debug_group11_contour_pixels.png";
        cv::imwrite(filename, visImage);
        std::cout << "  Saved contour pixel visualization: " << filename << std::endl;
    }
    // ========== END DEBUG VISUALIZATION ==========

    // Step 5: Launch tracing kernel (one block per group)
    dim3 blockSize(1, 1, 1); // Using single thread per block for now
    dim3 gridSize(numGroups, 1, 1);

    // Time the traceContoursParallel kernel
    gpuEvent_t traceStart, traceStop;
    gpuEventCreate(&traceStart);
    gpuEventCreate(&traceStop);
    gpuEventRecord(traceStart);

    traceContoursParallel_kernel<<<gridSize, blockSize>>>(
        contourBitmap,
        overlayBitmap,
        thrust::raw_pointer_cast(sortedPixels.indices.data()),
        thrust::raw_pointer_cast(sortedPixels.values.data()),
        thrust::raw_pointer_cast(d_unique_values.data()),
        thrust::raw_pointer_cast(d_group_starts.data()),
        thrust::raw_pointer_cast(d_group_counts.data()),
        numGroups,
        thrust::raw_pointer_cast(d_visited.data()),
        thrust::raw_pointer_cast(d_outputContours.data()),
        thrust::raw_pointer_cast(d_outputCounts.data()),
        thrust::raw_pointer_cast(d_subject_ids.data()),
        thrust::raw_pointer_cast(d_clipper_ids.data()),
        thrust::raw_pointer_cast(d_subject_counts.data()),
        thrust::raw_pointer_cast(d_clipper_counts.data()),
        width,
        height,
        maxPointsPerContour);

    CHECK_GPU_ERROR(gpuGetLastError());
    gpuEventRecord(traceStop);
    gpuEventSynchronize(traceStop);

    float traceMs = 0.0f;
    gpuEventElapsedTime(&traceMs, traceStart, traceStop);
    if (g_kernelProfiler) g_kernelProfiler->addTraceContoursTime(traceMs);

    gpuEventDestroy(traceStart);
    gpuEventDestroy(traceStop);

    // Step 6: Copy results back to host
    thrust::host_vector<uint2> h_outputContours = d_outputContours;
    thrust::host_vector<unsigned int> h_outputCounts = d_outputCounts;
    thrust::host_vector<unsigned int> h_subject_ids = d_subject_ids;
    thrust::host_vector<unsigned int> h_clipper_ids = d_clipper_ids;
    thrust::host_vector<unsigned int> h_subject_counts = d_subject_counts;
    thrust::host_vector<unsigned int> h_clipper_counts = d_clipper_counts;

    // Step 7: Convert to cv::Point format and print polygon IDs
    // Parse contours with delimiter markers (x=0xFFFFFFFF, y=pixel_count)
    for (unsigned int g = 0; g < numGroups; ++g) {
        unsigned int totalCount = h_outputCounts[g];
        if (totalCount > 0) {
            unsigned int baseIdx = g * maxPointsPerContour;
            unsigned int i = 0;
            unsigned int contourInGroup = 0;

            while (i < totalCount) {
                // Determine contour length
                unsigned int contourLen = 0;
                unsigned int startIdx = i;

                if (contourInGroup == 0) {
                    // First contour: find length by scanning until marker or end
                    while (i < totalCount) {
                        uint2 pt = h_outputContours[baseIdx + i];
                        if (pt.x == 0xFFFFFFFF) {
                            // Found marker for next contour
                            break;
                        }
                        i++;
                    }
                    contourLen = i - startIdx;
                } else {
                    // Subsequent contours: current position should be at marker
                    uint2 marker = h_outputContours[baseIdx + i];
                    if (marker.x == 0xFFFFFFFF) {
                        contourLen = marker.y;
                        i++; // Skip marker
                        startIdx = i;
                        i += contourLen; // Move past contour points
                    } else {
                        // Unexpected: not a marker, skip
                        i++;
                        continue;
                    }
                }

                // Extract contour points
                if (contourLen > 0) {
                    std::vector<cv::Point> contour;
                    contour.reserve(contourLen);

                    for (unsigned int j = 0; j < contourLen; ++j) {
                        uint2 pt = h_outputContours[baseIdx + startIdx + j];
                        contour.push_back(cv::Point(pt.x, pt.y));
                    }

                    contours.push_back(std::move(contour));

                    // Print collected polygon IDs for this contour
                    unsigned int subjectCount = h_subject_counts[g];
                    unsigned int clipperCount = h_clipper_counts[g];
                    std::cout << "  Group " << g << " Contour " << contourInGroup
                              << " (" << contourLen << " points): ";
                    std::cout << "Subject IDs [" << subjectCount << "]: ";
                    for (unsigned int k = 0; k < subjectCount; ++k) {
                        std::cout << h_subject_ids[g * maxIDsPerGroup + k];
                        if (k < subjectCount - 1) std::cout << ", ";
                    }
                    std::cout << " | Clipper IDs [" << clipperCount << "]: ";
                    for (unsigned int k = 0; k < clipperCount; ++k) {
                        std::cout << h_clipper_ids[g * maxIDsPerGroup + k];
                        if (k < clipperCount - 1) std::cout << ", ";
                    }
                    std::cout << std::endl;
                }

                contourInGroup++;
            }
        }
    }

    std::cout << "GPU tracing complete: extracted " << contours.size() << " contours" << std::endl;

    // ========== VISUALIZATION: Save GPU contour plot ==========
    // Create blank image for visualization
    cv::Mat visImage = cv::Mat::zeros(height, width, CV_8UC3);

    // Copy unique values to host for group ID lookup
    thrust::host_vector<unsigned int> h_unique_values(d_unique_values);

    // Draw each contour with color based on group ID
    unsigned int contourIdx = 0;
    for (unsigned int g = 0; g < numGroups; ++g) {
        unsigned int totalCount = h_outputCounts[g];
        if (totalCount > 0) {
            unsigned int baseIdx = g * maxPointsPerContour;
            unsigned int i = 0;
            unsigned int contourInGroup = 0;

            // Determine color based on group ID
            cv::Scalar color;
            if (g < 100) {
                color = cv::Scalar(0, 0, 255); // Red (BGR format)
            } else if (g < 200) {
                color = cv::Scalar(0, 255, 0); // Green
            } else {
                color = cv::Scalar(255, 0, 0); // Blue
            }

            while (i < totalCount) {
                // Determine contour length
                unsigned int contourLen = 0;
                unsigned int startIdx = i;

                if (contourInGroup == 0) {
                    // First contour: find length by scanning until marker or end
                    while (i < totalCount) {
                        uint2 pt = h_outputContours[baseIdx + i];
                        if (pt.x == 0xFFFFFFFF) {
                            // Found marker for next contour
                            break;
                        }
                        i++;
                    }
                    contourLen = i - startIdx;
                } else {
                    // Subsequent contours: current position should be at marker
                    uint2 marker = h_outputContours[baseIdx + i];
                    if (marker.x == 0xFFFFFFFF) {
                        contourLen = marker.y;
                        i++; // Skip marker
                        startIdx = i;
                        i += contourLen; // Move past contour points
                    } else {
                        // Unexpected: not a marker, skip
                        i++;
                        continue;
                    }
                }

                // Draw contour if it has points
                if (contourLen > 0) {
                    // Draw lines connecting the contour points
                    for (unsigned int j = 0; j < contourLen; ++j) {
                        uint2 pt1 = h_outputContours[baseIdx + startIdx + j];
                        uint2 pt2 = h_outputContours[baseIdx + startIdx + ((j + 1) % contourLen)];

                        cv::Point p1(pt1.x, pt1.y);
                        cv::Point p2(pt2.x, pt2.y);
                        cv::line(visImage, p1, p2, color, 1, cv::LINE_AA);
                    }

                    // Decompose group value to get subject_id and clipper_id
                    unsigned int groupValue = h_unique_values[g];
                    unsigned int subject_id = (groupValue & 0xFFFF) - 1; // Lower 16 bits, subtract 1
                    unsigned int clipper_id = ((groupValue >> 16) & 0xFFFF) - 1; // Upper 16 bits, subtract 1

                    // Get first point for label placement
                    uint2 firstPt = h_outputContours[baseIdx + startIdx];
                    cv::Point labelPos(firstPt.x + 5, firstPt.y - 5); // Offset slightly from point

                    // Create label text
                    std::string label = "(" + std::to_string(subject_id) + "," + std::to_string(clipper_id) + ")";

                    // Draw label without background (transparent)
                    cv::putText(visImage, label, labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.3,
                                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

                    contourIdx++;
                }

                contourInGroup++;
            }
        }
    }

    // Save visualization
    std::string outputFilename = "gpu_contour_tracing_visualization.png";
    cv::imwrite(outputFilename, visImage);
    std::cout << "Saved GPU contour visualization to: " << outputFilename << std::endl;
    std::cout << "  Color coding: RED (groups 0-99), GREEN (groups 100-199), BLUE (groups 200+)" << std::endl;
    std::cout << "  Labels show (subject_id, clipper_id) at first point of each contour" << std::endl;
    // ========== END VISUALIZATION ==========

    return contours;
}

// ============================================================================
// GPU Kernels for Contour Detection and Tracing
// ============================================================================

// Extract contour pixels from overlay bitmap
__global__ void extractContours_kernel(
    const unsigned int* inputBitmap,
    unsigned int* contourBitmap,
    const int width,
    const int height,
    const int chunkDim,
    OperationType opType)
{
    __shared__ unsigned int s_isContourPixel[32][32];
    s_isContourPixel[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    uint2 chunkStart = make_uint2(blockIdx.x * chunkDim, blockIdx.y * chunkDim);
    uint2 chunkEnd = make_uint2(chunkStart.x + chunkDim, chunkStart.y + chunkDim);
    uint2 chunkEndClamped = make_uint2(min(chunkEnd.x, width), min(chunkEnd.y, height));

    uint2 apronStart = make_uint2(chunkStart.x - 1, chunkStart.y - 1);
    uint2 apronEnd = make_uint2(chunkEnd.x + 1, chunkEnd.y + 1);

    uint2 apronStartClamped = make_uint2(max(apronStart.x, 0), max(apronStart.y, 0));
    uint2 apronEndClamped = make_uint2(min(apronEnd.x, width), min(apronEnd.y, height));

    int g_ix = apronStart.x + threadIdx.x;
    int g_iy = apronStart.y + threadIdx.y;
    int g_idx = g_iy * width + g_ix;

    if (g_ix >= apronStartClamped.x && g_ix < apronEndClamped.x &&
        g_iy >= apronStartClamped.y && g_iy < apronEndClamped.y) {

        unsigned int pixelValue = inputBitmap[g_idx];
        if (pixelValue > 0) {
            unsigned int subject_id = pixelValue & 0xFFFF;
            unsigned int clipper_id = (pixelValue >> 16) & 0xFFFF;

            switch (opType) {
                case OperationType::INTERSECTION:
                    if (subject_id > 0 && clipper_id > 0) {
                        s_isContourPixel[threadIdx.y][threadIdx.x] = pixelValue;
                    }
                    break;
                case OperationType::UNION:
                    if (subject_id > 0 || clipper_id > 0) {
                        s_isContourPixel[threadIdx.y][threadIdx.x] = pixelValue;
                    }
                    break;
                case OperationType::DIFFERENCE:
                    if (subject_id > 0 && clipper_id == 0) {
                        s_isContourPixel[threadIdx.y][threadIdx.x] = pixelValue;
                    }
                    break;
                case OperationType::XOR:
                    if ((subject_id > 0) != (clipper_id > 0)) {
                        s_isContourPixel[threadIdx.y][threadIdx.x] = pixelValue;
                    }
                    break;
                case OperationType::OFFSET:
                default:
                    s_isContourPixel[threadIdx.y][threadIdx.x] = pixelValue;
                    break;
            }
        }
    }

    __syncthreads();

    if (g_ix < chunkStart.x || g_ix >= chunkEndClamped.x ||
        g_iy < chunkStart.y || g_iy >= chunkEndClamped.y) {
        return;
    }

    // Check if pixel is contour (has non-ROI neighbors using 4-connectivity)
    if (s_isContourPixel[threadIdx.y][threadIdx.x] > 0) {
        if ((s_isContourPixel[threadIdx.y - 1][threadIdx.x] == 0) ||
            (s_isContourPixel[threadIdx.y][threadIdx.x - 1] == 0) ||
            (s_isContourPixel[threadIdx.y + 1][threadIdx.x] == 0) ||
            (s_isContourPixel[threadIdx.y][threadIdx.x + 1] == 0)) {

            contourBitmap[g_idx] = s_isContourPixel[threadIdx.y][threadIdx.x];
        }
    }
}

// ============================================================================
// Contour Tracing Kernel (Suzuki-Abe Algorithm) with Polygon ID Collection
// ============================================================================
//
// GPU-BASED CONTOUR SIMPLIFICATION ALGORITHM OVERVIEW
// =====================================================
//
// Context:
// --------
// For INTERSECTION operations, the overlay bitmap stores both subject and clipper
// polygon IDs in each pixel value:
//   - Lower 16 bits: Subject polygon ID
//   - Upper 16 bits: Clipper polygon ID
//
// The raw contour pixels (extracted by extractContours_kernel) contain pixels that
// belong to one or more subject/clipper polygon pairs. To simplify these raw contours,
// we need to:
//   1. Collect all subject and clipper IDs associated with each raw contour
//   2. Use these IDs to fetch relevant intersection points and polygon vertices
//   3. Match raw contour pixels to these geometrically significant points
//
// ID Collection During Tracing:
// -----------------------------
// During the traceOneContour() function, as we trace each pixel of a raw contour,
// we extract and collect the subject_id and clipper_id from the overlay bitmap pixel value.
//
// We maintain two small buffers (max 256 entries each) for unique IDs:
//   - subject_ids[256]: Stores unique subject polygon IDs
//   - clipper_ids[256]: Stores unique clipper polygon IDs
//   - subject_count: Number of unique subject IDs collected
//   - clipper_count: Number of unique clipper IDs collected
//
// As we trace, for each pixel:
//   1. Read pixel value from overlay bitmap (NOT contour bitmap)
//   2. Extract subject_id = pixel_value & 0xFFFF
//   3. Extract clipper_id = (pixel_value >> 16) & 0xFFFF
//   4. If subject_id > 0 and not already in buffer, add to subject_ids[]
//   5. If clipper_id > 0 and not already in buffer, add to clipper_ids[]
//
// Why This Works:
// ---------------
// - Contour pixels represent boundaries between intersecting regions
// - A single contour may touch multiple subject/clipper polygon pairs
// - Collecting ALL relevant IDs allows us to gather ALL candidate simplification points
// - The 256-entry limit is reasonable: most contours touch only a few polygons
//
// Next Step (Contour Simplification - TO BE IMPLEMENTED):
// -------------------------------------------------------
// After collecting IDs, we will:
//   1. For each subject_id + clipper_id pair, fetch intersection points from d_intersection_points
//   2. For each subject_id, fetch polygon vertices from subject layer
//   3. For each clipper_id, fetch polygon vertices from clipper layer
//   4. Create candidate point list with distance thresholds
//   5. Match raw contour pixels to candidate points
//   6. Keep only matched pixels as simplified contour
//
// ============================================================================

// Device helper: Get pixel at bitmap location
__device__ inline unsigned int getBitmapPixel(
    const unsigned int* bitmap,
    int x, int y,
    unsigned int width,
    unsigned int height)
{
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0;
    }
    return bitmap[y * width + x];
}

// Device helper: Find index in sortedIndices for a given bitmap index
__device__ inline int findSortedIndex(
    const unsigned int* sortedIndices,
    unsigned int targetIdx,
    unsigned int groupStart,
    unsigned int groupCount)
{
    for (unsigned int i = 0; i < groupCount; ++i) {
        if (sortedIndices[groupStart + i] == targetIdx) {
            return groupStart + i;
        }
    }
    return -1;
}

// Device helper: Add polygon ID to buffer if not already present
__device__ inline void addPolygonID(
    unsigned int* idBuffer,
    unsigned int* idCount,
    unsigned int newID,
    unsigned int maxIDs)
{
    if (newID == 0) return; // Skip zero IDs

    // Check if ID already exists
    for (unsigned int i = 0; i < *idCount; ++i) {
        if (idBuffer[i] == newID) {
            return; // Already in buffer
        }
    }

    // Add new ID if there's space
    if (*idCount < maxIDs) {
        idBuffer[*idCount] = newID;
        (*idCount)++;
    }
}

// Device function: Trace one contour using Suzuki-Abe 8-connectivity border following
// with polygon ID collection for intersection simplification
__device__ void traceOneContour(
    const unsigned int* contourBitmap,
    const unsigned int* overlayBitmap,
    const unsigned int* sortedIndices,
    unsigned int startSortedIdx,
    unsigned int targetValue,
    unsigned int groupStart,
    unsigned int groupCount,
    unsigned char* visited,
    uint2* outputContour,
    unsigned int* outputCount,
    unsigned int* subjectIDs,
    unsigned int* clipperIDs,
    unsigned int* subjectCount,
    unsigned int* clipperCount,
    unsigned int width,
    unsigned int height,
    unsigned int maxPoints,
    unsigned int groupIdx,
    unsigned int contourIdx)
{
    // Get starting pixel location
    unsigned int startIdx = sortedIndices[startSortedIdx];
    unsigned int startX = startIdx % width;
    unsigned int startY = startIdx / width;

    // Debug output for group 11 only
    bool debug = (groupIdx == 11);

    // Initialize polygon ID counts
    *subjectCount = 0;
    *clipperCount = 0;

    // Direction vectors for 8-connectivity (E, SE, S, SW, W, NW, N, NE)
    // Clockwise order starting from right (East)
    const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    unsigned int currentX = startX;
    unsigned int currentY = startY;
    int currentDir = 0; // Start looking right (East)

    unsigned int pointCount = 0;

    if (debug) {
        printf("# [Group %u, Contour %u] Starting trace at (%u, %u), targetValue=%u, startSortedIdx=%u\n",
               groupIdx, contourIdx, startX, startY, targetValue, startSortedIdx);
        printf("contour_%u_%u = [\n", groupIdx, contourIdx);
    }

    // Add starting point
    if (pointCount < maxPoints) {
        outputContour[pointCount].x = currentX;
        outputContour[pointCount].y = currentY;
        pointCount++;
        if (debug) {
            printf("    (%u, %u),  # Point %u | sortedIdx=%u, wasVisited=%d\n",
                   currentX, currentY, pointCount - 1, startSortedIdx,
                   visited[startSortedIdx] != 0 ? 1 : 0);
        }
    }

    // Collect polygon IDs from starting point
    unsigned int overlayValue = getBitmapPixel(overlayBitmap, currentX, currentY, width, height);
    unsigned int subject_id = overlayValue & 0xFFFF;        // Lower 16 bits
    unsigned int clipper_id = (overlayValue >> 16) & 0xFFFF; // Upper 16 bits
    addPolygonID(subjectIDs, subjectCount, subject_id, 256);
    addPolygonID(clipperIDs, clipperCount, clipper_id, 256);

    // Mark as visited
    visited[startSortedIdx] = 1;

    // Border following loop
    bool firstMove = true;
    unsigned int iterCount = 0;
    const unsigned int maxIter = width * height * 4; // Safety limit

    while (iterCount < maxIter) {
        iterCount++;

        // Search for next contour pixel in clockwise order
        // For 8-connectivity, start from 2 positions counter-clockwise
        int searchDir = (currentDir + 6) % 8; // Start searching counter-clockwise
        bool found = false;

        for (int i = 0; i < 8; ++i) {
            int checkDir = (searchDir + i) % 8;
            int nextX = currentX + dx[checkDir];
            int nextY = currentY + dy[checkDir];

            // Check if this pixel is part of the contour
            unsigned int pixelValue = getBitmapPixel(
                contourBitmap, nextX, nextY, width, height);

            if (pixelValue == targetValue) {
                // Check visited status BEFORE accepting this pixel
                unsigned int nextIdx = nextY * width + nextX;
                int sortedIdx = findSortedIndex(
                    sortedIndices, nextIdx, groupStart, groupCount);
                bool wasVisited = (sortedIdx >= 0) ? (visited[sortedIdx] != 0) : false;

                // Skip already-visited pixels (except when returning to start to close contour)
                if (wasVisited && !(nextX == startX && nextY == startY)) {
                    if (debug) {
                        printf("# Skipping already-visited pixel (%d, %d) at direction %d\n",
                               nextX, nextY, checkDir);
                    }
                    continue; // Skip this pixel and try next direction
                }

                // Found next contour pixel (unvisited or returning to start)
                currentX = nextX;
                currentY = nextY;
                currentDir = checkDir;
                found = true;

                // Check if we've returned to start
                if (!firstMove && currentX == startX && currentY == startY) {
                    // Closed contour
                    *outputCount = pointCount;
                    if (debug) {
                        printf("]\n# [Group %u, Contour %u] Closed contour with %u points\n\n",
                               groupIdx, contourIdx, pointCount);
                    }
                    return;
                }
                firstMove = false;

                // Add point to contour
                if (pointCount < maxPoints) {
                    outputContour[pointCount].x = currentX;
                    outputContour[pointCount].y = currentY;
                    if (debug) {
                        printf("    (%u, %u),  # Point %u | sortedIdx=%d, wasVisited=%d\n",
                               currentX, currentY, pointCount, sortedIdx, wasVisited ? 1 : 0);
                    }
                    pointCount++;
                }

                // Collect polygon IDs from current point
                unsigned int overlayValue = getBitmapPixel(overlayBitmap, currentX, currentY, width, height);
                unsigned int subject_id = overlayValue & 0xFFFF;        // Lower 16 bits
                unsigned int clipper_id = (overlayValue >> 16) & 0xFFFF; // Upper 16 bits
                addPolygonID(subjectIDs, subjectCount, subject_id, 256);
                addPolygonID(clipperIDs, clipperCount, clipper_id, 256);

                // Mark as visited
                if (sortedIdx >= 0) {
                    visited[sortedIdx] = 1;
                }

                break;
            }
        }

        if (!found) {
            // No neighbor found, end tracing
            if (debug) {
                printf("]\n# [Group %u, Contour %u] No neighbor found, ending trace with %u points (OPEN contour)\n\n",
                       groupIdx, contourIdx, pointCount);
            }
            break;
        }
    }// while (iterCount < maxIter)

    *outputCount = pointCount;

    // If we exited the loop without printing the closing bracket (max iterations reached)
    if (debug && pointCount > 0 && iterCount >= maxIter) {
        printf("]\n# [Group %u, Contour %u] Max iterations reached, trace incomplete with %u points\n\n",
               groupIdx, contourIdx, pointCount);
    }
}

// Main kernel: Each block processes one group
__global__ void traceContoursParallel_kernel(
    const unsigned int* contourBitmap,
    const unsigned int* overlayBitmap,
    const unsigned int* sortedIndices,
    const unsigned int* sortedValues,
    const unsigned int* groupPixelValues,
    const unsigned int* groupStartIndices,
    const unsigned int* groupCounts,
    const unsigned int numGroups,
    unsigned char* visited,
    uint2* outputContours,
    unsigned int* outputCounts,
    unsigned int* outputSubjectIDs,
    unsigned int* outputClipperIDs,
    unsigned int* outputSubjectCounts,
    unsigned int* outputClipperCounts,
    const unsigned int width,
    const unsigned int height,
    const unsigned int maxPointsPerContour)
{
    unsigned int groupIdx = blockIdx.x;

    if (groupIdx >= numGroups) {
        return;
    }

    // Get group info from separate arrays
    unsigned int targetValue = groupPixelValues[groupIdx];
    unsigned int groupStart = groupStartIndices[groupIdx];
    unsigned int groupCount = groupCounts[groupIdx];

    // Each thread in the block tries to find an unvisited pixel to start tracing
    // This handles multiple disjoint contours within the same group
    unsigned int contourCount = 0;
    const unsigned int maxContoursPerGroup = 32; // Safety limit
    const unsigned int maxIDsPerGroup = 256;

    // Allocate shared memory for polygon ID collection
    __shared__ unsigned int sharedSubjectIDs[256];
    __shared__ unsigned int sharedClipperIDs[256];
    __shared__ unsigned int sharedSubjectCount;
    __shared__ unsigned int sharedClipperCount;

    // Only thread 0 does the tracing (serial within block for now)
    if (threadIdx.x == 0) {
        // Scan through all pixels in this group
        for (unsigned int i = 0; i < groupCount && contourCount < maxContoursPerGroup; ++i) {
            unsigned int sortedIdx = groupStart + i;

            // Check if already visited
            if (visited[sortedIdx] == 0) {
                // Start a new contour from this pixel
                unsigned int localCount = 0;
                uint2* contourOutput = outputContours + groupIdx * maxPointsPerContour;
                unsigned int currentOffset = outputCounts[groupIdx];

                // For second and subsequent contours, add a delimiter marker first
                // Marker format: x = 0xFFFFFFFF, y = will be filled with pixel count after tracing
                unsigned int markerIdx = 0;
                if (contourCount > 0) {
                    // Check if we have space for marker
                    if (currentOffset >= maxPointsPerContour - 1) {
                        break; // No space left
                    }
                    markerIdx = currentOffset;
                    contourOutput[currentOffset].x = 0xFFFFFFFF;
                    contourOutput[currentOffset].y = 0; // Will be updated after tracing
                    currentOffset++;
                    outputCounts[groupIdx]++;
                }

                traceOneContour(
                    contourBitmap,
                    overlayBitmap,
                    sortedIndices,
                    sortedIdx,
                    targetValue,
                    groupStart,
                    groupCount,
                    visited,
                    contourOutput + currentOffset,
                    &localCount,
                    sharedSubjectIDs,
                    sharedClipperIDs,
                    &sharedSubjectCount,
                    &sharedClipperCount,
                    width,
                    height,
                    maxPointsPerContour - currentOffset,
                    groupIdx,
                    contourCount);

                // Store the collected polygon IDs for this group
                // For now, each group gets one set of IDs (could be refined later)
                if (contourCount == 0) {
                    // First contour: copy shared memory IDs to global memory
                    for (unsigned int j = 0; j < sharedSubjectCount && j < maxIDsPerGroup; ++j) {
                        outputSubjectIDs[groupIdx * maxIDsPerGroup + j] = sharedSubjectIDs[j];
                    }
                    for (unsigned int j = 0; j < sharedClipperCount && j < maxIDsPerGroup; ++j) {
                        outputClipperIDs[groupIdx * maxIDsPerGroup + j] = sharedClipperIDs[j];
                    }
                    outputSubjectCounts[groupIdx] = sharedSubjectCount;
                    outputClipperCounts[groupIdx] = sharedClipperCount;
                } else {
                    // For subsequent contours, update the marker's y value with pixel count
                    contourOutput[markerIdx].y = localCount;
                }

                outputCounts[groupIdx] += localCount;
                contourCount++;
            }
        }
    }
}

} // namespace GpuLithoLib
