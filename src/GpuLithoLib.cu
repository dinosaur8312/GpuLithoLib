#include "../include/GpuLithoLib.h"
#include "LayerImpl.h"
#include "GpuOperations.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <memory>
#include <set>
#include <map>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/iterator/transform_iterator.h>

namespace GpuLithoLib {

using gpuLitho::OperationType;

// Functor for filtering intersection pixels (both subject_id and clipper_id non-zero)
struct IsIntersectionPixel {
    __host__ __device__
    bool operator()(unsigned int pixel_value) const {
        unsigned int subject_id = pixel_value & 0xFFFF;
        unsigned int clipper_id = (pixel_value >> 16) & 0xFFFF;
        return (subject_id > 0) && (clipper_id > 0);
    }
};

// === Intersection Point Computation ===

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

// Structure for candidate points in contour simplification (matches main.cu)
struct CandidatePoint {
    unsigned int x;
    unsigned int y;
    float max_distance_threshold;
    
    CandidatePoint(unsigned int px, unsigned int py, float threshold)
        : x(px), y(py), max_distance_threshold(threshold) {}
};

// Point types for intersection tracking
enum class PointType {
    REAL_INTERSECTION,  // Actual edge-edge intersection
    SUBJECT_VERTEX,     // Vertex from subject polygon
    CLIPPER_VERTEX      // Vertex from clipper polygon
};

// Structure to represent an intersection point with tracking information
struct IntersectionPoint {
    std::pair<unsigned int, unsigned int> position;  // x, y coordinates
    bool isMatched;                                  // Track if used during simplification
    float max_distance_threshold;                    // Maximum distance for matching based on intersection angle
    PointType type;                                  // Type of point
    
    // Constructors
    IntersectionPoint(unsigned int x, unsigned int y, float threshold = 2.0f, 
                     PointType pt = PointType::REAL_INTERSECTION) 
        : position(x, y), isMatched(false), max_distance_threshold(threshold), type(pt) {}
    
    IntersectionPoint(const std::pair<unsigned int, unsigned int>& pos, float threshold = 2.0f,
                     PointType pt = PointType::REAL_INTERSECTION) 
        : position(pos), isMatched(false), max_distance_threshold(threshold), type(pt) {}
    
    // Comparator for use in std::set
    bool operator<(const IntersectionPoint& other) const {
        if (position.first != other.position.first) 
            return position.first < other.position.first;
        return position.second < other.position.second;
    }
    
    bool operator==(const IntersectionPoint& other) const {
        return position.first == other.position.first && 
               position.second == other.position.second;
    }
};

//==============================================================================
// Layer implementation
//==============================================================================

Layer::Layer() : impl(std::make_unique<LayerImpl>()) {}

Layer::Layer(const Layer& other) : impl(std::make_unique<LayerImpl>(*other.impl)) {}

Layer& Layer::operator=(const Layer& other) {
    if (this != &other) {
        impl = std::make_unique<LayerImpl>(*other.impl);
    }
    return *this;
}

Layer::Layer(Layer&& other) noexcept : impl(std::move(other.impl)) {}

Layer& Layer::operator=(Layer&& other) noexcept {
    if (this != &other) {
        impl = std::move(other.impl);
    }
    return *this;
}

Layer::~Layer() = default;

Layer::Layer(std::unique_ptr<LayerImpl> impl) : impl(std::move(impl)) {}

bool Layer::empty() const {
    return impl ? impl->empty() : true;
}

unsigned int Layer::getPolygonCount() const {
    return impl ? impl->polygonCount : 0;
}

unsigned int Layer::getVertexCount() const {
    return impl ? impl->vertexCount : 0;
}

std::vector<unsigned int> Layer::getBoundingBox() const {
    return impl ? impl->getBoundingBox() : std::vector<unsigned int>{0, 0, 0, 0};
}

const uint2* Layer::getHostVertices() const {
    return impl ? impl->h_vertices : nullptr;
}

const unsigned int* Layer::getHostStartIndices() const {
    return impl ? impl->h_startIndices : nullptr;
}

const unsigned int* Layer::getHostPtCounts() const {
    return impl ? impl->h_ptCounts : nullptr;
}

const uint2* Layer::getDeviceVertices() const {
    return impl ? impl->d_vertices : nullptr;
}

const unsigned int* Layer::getDeviceStartIndices() const {
    return impl ? impl->d_startIndices : nullptr;
}

const unsigned int* Layer::getDevicePtCounts() const {
    return impl ? impl->d_ptCounts : nullptr;
}

//==============================================================================
// GpuLithoEngine implementation
//==============================================================================

class GpuLithoEngine::EngineImpl {
public:
    // Maximum working area size
    unsigned int maxGridWidth;
    unsigned int maxGridHeight;
    
    // Current actual grid size (set during layer preparation)
    unsigned int currentGridWidth;
    unsigned int currentGridHeight;
    
    // Global bounding box in original coordinates
    bool isGlobalBoxSet;
    unsigned int globalMinX, globalMinY, globalMaxX, globalMaxY;
    int shiftX, shiftY;  // Shift amounts applied to normalize to origin
    
    // Prepared layers (stored as shifted copies for operations)
    std::unique_ptr<LayerImpl> preparedLayer1;
    std::unique_ptr<LayerImpl> preparedLayer2;
    
    bool profilingEnabled;
    
    // Performance tracking
    float totalRayCastingTime;
    float totalOverlayTime;
    float totalContourTime;
    int operationCount;
    
    EngineImpl(unsigned int maxW, unsigned int maxH) 
        : maxGridWidth(maxW), maxGridHeight(maxH),
          currentGridWidth(0), currentGridHeight(0),
          isGlobalBoxSet(false), globalMinX(0), globalMinY(0), globalMaxX(0), globalMaxY(0),
          shiftX(0), shiftY(0), profilingEnabled(false),
          totalRayCastingTime(0), totalOverlayTime(0), totalContourTime(0), operationCount(0) {}
    
    // Prepare engine for single layer operation
    void prepareSingleLayer(const Layer& layer) {
        if (!layer.impl || layer.impl->empty()) {
            std::cerr << "Error: Cannot prepare with empty layer" << std::endl;
            return;
        }
        
        // Calculate bounding box
        auto bbox = layer.impl->getBoundingBox();
        globalMinX = bbox[0];
        globalMinY = bbox[1]; 
        globalMaxX = bbox[2];
        globalMaxY = bbox[3];
        
        // Add padding
        if (globalMinX > 0) globalMinX--;
        if (globalMinY > 0) globalMinY--;
        globalMaxX++;
        globalMaxY++;
        
        // Calculate shift amounts to normalize to origin
        shiftX = -static_cast<int>(globalMinX);
        shiftY = -static_cast<int>(globalMinY);
        
        // Set current grid size
        currentGridWidth = globalMaxX - globalMinX;
        currentGridHeight = globalMaxY - globalMinY;
        
        // Check bounds
        if (currentGridWidth > maxGridWidth || currentGridHeight > maxGridHeight) {
            std::cerr << "Warning: Layer size (" << currentGridWidth << "x" << currentGridHeight 
                      << ") exceeds maximum (" << maxGridWidth << "x" << maxGridHeight << ")" << std::endl;
        }
        
        // Create shifted copy of layer
        preparedLayer1 = std::make_unique<LayerImpl>(*layer.impl);
        preparedLayer1->shift(shiftX, shiftY);
        
        preparedLayer2.reset();  // Clear second layer
        isGlobalBoxSet = true;
        
        if (profilingEnabled) {
            std::cout << "Prepared single layer: global box (" << globalMinX << "," << globalMinY 
                      << "," << globalMaxX << "," << globalMaxY << "), grid size " 
                      << currentGridWidth << "x" << currentGridHeight << std::endl;
        }
    }
    
    // Prepare engine for dual layer operation
    void prepareDualLayers(const Layer& layer1, const Layer& layer2) {
        if (!layer1.impl || layer1.impl->empty() || !layer2.impl || layer2.impl->empty()) {
            std::cerr << "Error: Cannot prepare with empty layers" << std::endl;
            return;
        }
        
        // Calculate combined bounding box
        auto bbox1 = layer1.impl->getBoundingBox();
        auto bbox2 = layer2.impl->getBoundingBox();
        
        globalMinX = std::min(bbox1[0], bbox2[0]);
        globalMinY = std::min(bbox1[1], bbox2[1]);
        globalMaxX = std::max(bbox1[2], bbox2[2]);
        globalMaxY = std::max(bbox1[3], bbox2[3]);
        
        // Add padding
        if (globalMinX > 0) globalMinX--;
        if (globalMinY > 0) globalMinY--;
        globalMaxX++;
        globalMaxY++;
        
        // Calculate shift amounts
        shiftX = -static_cast<int>(globalMinX);
        shiftY = -static_cast<int>(globalMinY);
        
        // Set current grid size
        currentGridWidth = globalMaxX - globalMinX;
        currentGridHeight = globalMaxY - globalMinY;
        
        // Check bounds
        if (currentGridWidth > maxGridWidth || currentGridHeight > maxGridHeight) {
            std::cerr << "Warning: Combined layer size (" << currentGridWidth << "x" << currentGridHeight 
                      << ") exceeds maximum (" << maxGridWidth << "x" << maxGridHeight << ")" << std::endl;
        }
        
        // Create shifted copies
        preparedLayer1 = std::make_unique<LayerImpl>(*layer1.impl);
        preparedLayer1->shift(shiftX, shiftY);
        
        preparedLayer2 = std::make_unique<LayerImpl>(*layer2.impl);
        preparedLayer2->shift(shiftX, shiftY);
        
        isGlobalBoxSet = true;
        
        if (profilingEnabled) {
            std::cout << "Prepared dual layers: global box (" << globalMinX << "," << globalMinY 
                      << "," << globalMaxX << "," << globalMaxY << "), grid size " 
                      << currentGridWidth << "x" << currentGridHeight << std::endl;
        }
    }
    
    // Reset engine state
    void reset() {
        isGlobalBoxSet = false;
        currentGridWidth = currentGridHeight = 0;
        globalMinX = globalMinY = globalMaxX = globalMaxY = 0;
        shiftX = shiftY = 0;
        preparedLayer1.reset();
        preparedLayer2.reset();
    }
    
    // Restore original coordinates to output layer
    void restoreOriginalCoordinates(LayerImpl* layer) {
        if (layer && isGlobalBoxSet) {
            layer->shift(-shiftX, -shiftY);  // Reverse the normalization shift
        }
    }
    
    // Perform ray casting on a layer
    void performRayCasting(LayerImpl* layer, int edgeMode = 1) {
        if (!layer || layer->empty()) return;
        
        layer->ensureBitmapAllocated(currentGridWidth, currentGridHeight);
        layer->copyToDevice();
        layer->calculateBoundingBoxes();
        
        gpuEvent_t start, stop;
        if (profilingEnabled) {
            gpuEventCreate(&start);
            gpuEventCreate(&stop);
            gpuEventRecord(start);
        }
        
        // Launch ray casting kernel
        rayCasting_kernel<<<layer->polygonCount, 512>>>(
            layer->d_vertices,
            layer->d_startIndices, 
            layer->d_ptCounts,
            layer->d_boxes,
            layer->d_bitmap,
            currentGridWidth,
            currentGridHeight,
            layer->polygonCount);
        
        CHECK_GPU_ERROR(gpuGetLastError());
        
        // Optionally render edges
        if (edgeMode >= 0) {
            edgeRender_kernel<<<layer->polygonCount, 256>>>(
                layer->d_vertices,
                layer->d_startIndices,
                layer->d_ptCounts,
                layer->d_bitmap,
                currentGridWidth,
                currentGridHeight,
                edgeMode);
            
            CHECK_GPU_ERROR(gpuGetLastError());
        }
        
        CHECK_GPU_ERROR(gpuDeviceSynchronize());
        
        if (profilingEnabled) {
            gpuEventRecord(stop);
            gpuEventSynchronize(stop);
            float elapsed;
            gpuEventElapsedTime(&elapsed, start, stop);
            totalRayCastingTime += elapsed;
            gpuEventDestroy(start);
            gpuEventDestroy(stop);
        }
    }
    
    // Perform overlay operation
    void performOverlay(LayerImpl* subject, LayerImpl* clipper, LayerImpl* output) {
        if (!subject || !clipper || !output) return;
        
        output->ensureBitmapAllocated(currentGridWidth, currentGridHeight);
        
        gpuEvent_t start, stop;
        if (profilingEnabled) {
            gpuEventCreate(&start);
            gpuEventCreate(&stop);
            gpuEventRecord(start);
        }
        
        int blocksPerGrid = currentGridHeight;
        int threadsPerBlock = 512;
        
        overlay_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            subject->d_bitmap,
            clipper->d_bitmap,
            output->d_bitmap,
            currentGridWidth,
            currentGridHeight);
        
        CHECK_GPU_ERROR(gpuGetLastError());
        CHECK_GPU_ERROR(gpuDeviceSynchronize());
        
        if (profilingEnabled) {
            gpuEventRecord(stop);
            gpuEventSynchronize(stop);
            float elapsed;
            gpuEventElapsedTime(&elapsed, start, stop);
            totalOverlayTime += elapsed;
            gpuEventDestroy(start);
            gpuEventDestroy(stop);
        }
    }
    
    // IMPORTANT: This is a simplified contour extraction for visualization purposes only.
    // The correct implementation should follow this pipeline:
    // 1. extractIntersectingPolygonPairs(outputLayer) -> get intersecting polygon pairs
    // 2. computeIntersectionPoints(pairs, subject, clipper) -> get precise intersection points 
    // 3. detectContours(outputLayer) -> get raw contours from bitmap
    // 4. simplifyContours(raw_contours, vertex_data, intersection_points) -> final simplified contours
    // 
    // This simplified version only does steps 3-4 without intersection point analysis,
    // so the results are not geometrically accurate for actual lithography use.
    Layer extractContours(LayerImpl* input, OperationType opType) {
        if (!input) return Layer();
        
        auto contourLayer = std::make_unique<LayerImpl>();
        contourLayer->ensureBitmapAllocated(currentGridWidth, currentGridHeight);
        contourLayer->clearBitmap();
        
        gpuEvent_t start, stop;
        if (profilingEnabled) {
            gpuEventCreate(&start);
            gpuEventCreate(&stop);
            gpuEventRecord(start);
        }
        
        // Extract contours using GPU
        const int chunkDim = 30;
        dim3 blockSize(32, 32);
        dim3 gridSize(iDivUp(currentGridWidth, chunkDim), iDivUp(currentGridHeight, chunkDim));
        
        extractContours_kernel<<<gridSize, blockSize>>>(
            input->d_bitmap,
            contourLayer->d_bitmap,
            currentGridWidth,
            currentGridHeight,
            chunkDim,
            opType);
        
        CHECK_GPU_ERROR(gpuGetLastError());
        CHECK_GPU_ERROR(gpuDeviceSynchronize());
        
        // Copy bitmap to host for contour detection
        contourLayer->copyBitmapToHost();
        
        // Use OpenCV to find contours
        cv::Mat binaryImage(currentGridHeight, currentGridWidth, CV_8UC1);
        for (int y = 0; y < currentGridHeight; ++y) {
            for (int x = 0; x < currentGridWidth; ++x) {
                int idx = y * currentGridWidth + x;
                binaryImage.at<uchar>(y, x) = contourLayer->h_bitmap[idx] > 0 ? 255 : 0;
            }
        }
        
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binaryImage, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
        
        // Convert contours back to polygon format
        for (const auto& contour : contours) {
            if (contour.size() >= 3) {
                std::vector<uint2> vertices;
                for (const auto& pt : contour) {
                    vertices.push_back({static_cast<unsigned int>(pt.x), static_cast<unsigned int>(pt.y)});
                }
                contourLayer->addPolygon(vertices.data(), vertices.size());
            }
        }
        
        // CRITICAL: Restore original coordinates before returning
        restoreOriginalCoordinates(contourLayer.get());
        
        if (profilingEnabled) {
            gpuEventRecord(stop);
            gpuEventSynchronize(stop);
            float elapsed;
            gpuEventElapsedTime(&elapsed, start, stop);
            totalContourTime += elapsed;
            operationCount++;
            gpuEventDestroy(start);
            gpuEventDestroy(stop);
        }
        
        return Layer(std::move(contourLayer));
    }
    
    // Compute all intersection points from output layer (combines extractIntersectingPolygonPairs + computeIntersectionPoints)
    // Input: outputLayer after overlay operation
    // Output: map of polygon pairs to their intersection points
    std::map<std::pair<unsigned int, unsigned int>, std::set<IntersectionPoint>>
    computeAllIntersectionPoints(LayerImpl* outputLayer, LayerImpl* subjectLayer, LayerImpl* clipperLayer) {
        std::map<std::pair<unsigned int, unsigned int>, std::set<IntersectionPoint>> intersection_points_set;

        if (!outputLayer || !subjectLayer || !clipperLayer) {
            std::cerr << "Error: Null layer in computeAllIntersectionPoints" << std::endl;
            return intersection_points_set;
        }

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
        thrust::device_vector<unsigned int> d_packed_pairs(totalPixels);

        auto compact_end = thrust::copy_if(
            d_bitmap_ptr,                    // First iterator: start of bitmap
            d_bitmap_ptr + totalPixels,      // Last iterator: end of bitmap
            d_packed_pairs.begin(),          // Output destination
            IsIntersectionPixel()            // Predicate functor
        );
    
        // Resize to actual compacted size (remove unused capacity)
        auto compact_size = static_cast<size_t>(compact_end - d_packed_pairs.begin());
        d_packed_pairs.resize(compact_size);

        // Step 1b: Sort using radix sort (optimal for GPUs)
        thrust::sort(d_packed_pairs.begin(), d_packed_pairs.end());

        // Step 1c: Remove duplicates to get unique pairs
        auto unique_end = thrust::unique(d_packed_pairs.begin(), d_packed_pairs.end());
        auto unique_size = static_cast<size_t>(unique_end - d_packed_pairs.begin());
        d_packed_pairs.resize(unique_size);

        // Step 1d: Copy unique pairs back to host (minimal transfer - only unique pairs)
        std::vector<unsigned int> h_packed_pairs(d_packed_pairs.size());
        thrust::copy(d_packed_pairs.begin(), d_packed_pairs.end(), h_packed_pairs.begin());

        // Step 2: Compute intersection points for each intersecting polygon pair using GPU
        const unsigned int MAX_INTERSECTIONS_PER_PAIR = 32;
        unsigned int numPairs = d_packed_pairs.size();

        if (numPairs > 0) {
            // Allocate device memory for intersection points output
            // 2D array: [numPairs][32] intersection points
            thrust::device_vector<IntersectionPointData> d_intersection_points(numPairs * MAX_INTERSECTIONS_PER_PAIR);
            thrust::device_vector<unsigned int> d_intersection_counts(numPairs);

            // Launch kernel: one block per intersecting pair
            dim3 blockSize(256);  // 256 threads per block
            dim3 gridSize(numPairs);

            computeIntersectionPoints_kernel<<<gridSize, blockSize>>>(
                thrust::raw_pointer_cast(d_packed_pairs.data()),
                numPairs,
                subjectLayer->d_vertices,
                subjectLayer->d_startIndices,
                subjectLayer->d_ptCounts,
                clipperLayer->d_vertices,
                clipperLayer->d_startIndices,
                clipperLayer->d_ptCounts,
                thrust::raw_pointer_cast(d_intersection_points.data()),
                thrust::raw_pointer_cast(d_intersection_counts.data()));

            CHECK_GPU_ERROR(gpuGetLastError());
            CHECK_GPU_ERROR(gpuDeviceSynchronize());

            // Copy results back to host
            std::vector<IntersectionPointData> h_intersection_points(d_intersection_points.size());
            std::vector<unsigned int> h_intersection_counts(d_intersection_counts.size());

            thrust::copy(d_intersection_points.begin(), d_intersection_points.end(), h_intersection_points.begin());
            thrust::copy(d_intersection_counts.begin(), d_intersection_counts.end(), h_intersection_counts.begin());

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
    
    // Detect raw contours from output layer bitmap
    std::vector<std::vector<cv::Point>> detectRawContours(LayerImpl* outputLayer, OperationType opType) {
        std::vector<std::vector<cv::Point>> contours;

        if (!outputLayer) {
            return contours;
        }

        // NOTE: The GPU kernel approach is commented out for future development
        // Currently using direct bitmap processing for correctness
        /*
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
        */

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

        return contours;
    }
    
    // Simplify contours using layer vertex information and intersection points
    // Input: raw contours, two layers, intersection points set, operation type
    // Output: geometrically accurate simplified contours
    // Implements the exact algorithm from main.cu lines 1283-1624
    std::vector<std::vector<cv::Point>> simplifyContoursWithGeometry(
        const std::vector<std::vector<cv::Point>>& raw_contours,
        LayerImpl* subjectLayer,
        LayerImpl* clipperLayer,
        LayerImpl* outputLayer,
        const std::map<std::pair<unsigned int, unsigned int>, std::set<IntersectionPoint>>& intersection_points_set,
        OperationType opType) {
        
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
                                    intersection_point.position.first,
                                    intersection_point.position.second,
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
    
    // Perform boolean operation with full pipeline
    Layer performBooleanOperation(OperationType opType) {
        if (!preparedLayer1 || !preparedLayer2) {
            std::cerr << "Error: Layers not prepared for boolean operation" << std::endl;
            return Layer();
        }
        
        auto output = std::make_unique<LayerImpl>();
        
        // Step 1: Ray casting for both layers
        performRayCasting(preparedLayer1.get(), (opType == OperationType::DIFFERENCE) ? 1 : 1);
        performRayCasting(preparedLayer2.get(), (opType == OperationType::DIFFERENCE) ? 0 : 1);
        
        // Step 2: Overlay
        performOverlay(preparedLayer1.get(), preparedLayer2.get(), output.get());
        
        // Step 3: Compute intersection points from overlay result
        auto intersection_points_set = computeAllIntersectionPoints(
            output.get(), preparedLayer1.get(), preparedLayer2.get());
        
        // Step 4: Detect raw contours
        auto raw_contours = detectRawContours(output.get(), opType);

        // Debug: Save raw contours visualization
        if (!raw_contours.empty()) {
            cv::Mat rawContourImage(currentGridHeight, currentGridWidth, CV_8UC3, cv::Scalar(255, 255, 255));
            for (const auto& contour : raw_contours) {
                if (contour.size() >= 2) {
                    for (size_t i = 0; i < contour.size(); ++i) {
                        size_t next_i = (i + 1) % contour.size();
                        cv::line(rawContourImage, contour[i], contour[next_i], cv::Scalar(0, 0, 255), 1);
                    }
                }
            }

            std::string debugRawFilename = "debug_raw_contours_";
            switch(opType) {
                case OperationType::INTERSECTION:
                    debugRawFilename += "intersection.png";
                    break;
                case OperationType::UNION:
                    debugRawFilename += "union.png";
                    break;
                case OperationType::DIFFERENCE:
                    debugRawFilename += "difference.png";
                    break;
                case OperationType::XOR:
                    debugRawFilename += "xor.png";
                    break;
                default:
                    debugRawFilename += "unknown.png";
                    break;
            }
            cv::imwrite(debugRawFilename, rawContourImage);
        }

        // Step 5: Simplify contours using geometry
        auto simplified_contours = simplifyContoursWithGeometry(
            raw_contours, preparedLayer1.get(), preparedLayer2.get(), 
            output.get(), intersection_points_set, opType);
        
        // Step 6: Convert simplified contours to Layer
        auto resultLayer = std::make_unique<LayerImpl>();
        for (const auto& contour : simplified_contours) {
            if (contour.size() >= 3) {
                std::vector<uint2> vertices;
                for (const auto& pt : contour) {
                    vertices.push_back({static_cast<unsigned int>(pt.x), 
                                      static_cast<unsigned int>(pt.y)});
                }
                resultLayer->addPolygon(vertices.data(), vertices.size());
            }
        }
        
        // Step 7: Restore original coordinates
        restoreOriginalCoordinates(resultLayer.get());
        
        return Layer(std::move(resultLayer));
    }
    
    // Perform single layer operation (uses prepared layer)
    Layer performSingleLayerOperation(OperationType opType, int offsetDistance = 0) {
        if (!preparedLayer1) {
            std::cerr << "Error: Layer not prepared for single layer operation" << std::endl;
            return Layer();
        }
        
        if (opType == OperationType::OFFSET) {
            // Handle offset operation
            auto offsetLayer = std::make_unique<LayerImpl>(*preparedLayer1);
            offsetLayer->ensureBitmapAllocated(currentGridWidth, currentGridHeight);
            
            // First, render the layer to bitmap
            performRayCasting(offsetLayer.get(), 1);
            
            // Apply offset using morphological operation
            bool positiveOffset = offsetDistance > 0;
            int absOffset = abs(offsetDistance);
            
            auto tempLayer = std::make_unique<LayerImpl>(*offsetLayer);
            
            dim3 blockDim(256);
            dim3 gridDim(iDivUp(currentGridWidth * currentGridHeight, blockDim.x));
            
            offset_kernel<<<gridDim, blockDim>>>(
                offsetLayer->d_vertices,
                offsetLayer->d_startIndices,
                offsetLayer->d_ptCounts,
                tempLayer->d_bitmap,
                currentGridWidth,
                currentGridHeight,
                absOffset,
                positiveOffset);
            
            CHECK_GPU_ERROR(gpuGetLastError());
            CHECK_GPU_ERROR(gpuDeviceSynchronize());
            
            // Extract contours from offset result
            return extractContours(tempLayer.get(), OperationType::OFFSET);
        }
        
        return Layer(); // Unsupported single layer operation
    }
};

//==============================================================================
// GpuLithoEngine public interface
//==============================================================================

GpuLithoEngine::GpuLithoEngine(unsigned int maxGridWidth, unsigned int maxGridHeight)
    : impl(std::make_unique<EngineImpl>(maxGridWidth, maxGridHeight)) {}

GpuLithoEngine::~GpuLithoEngine() = default;

void GpuLithoEngine::prepareSingleLayer(const Layer& layer) {
    impl->prepareSingleLayer(layer);
}

void GpuLithoEngine::prepareDualLayers(const Layer& layer1, const Layer& layer2) {
    impl->prepareDualLayers(layer1, layer2);
}

void GpuLithoEngine::reset() {
    impl->reset();
}

Layer GpuLithoEngine::createLayerFromFile(const std::string& filename, unsigned int layerIndex) {
    auto layerImpl = std::make_unique<LayerImpl>();
    if (layerImpl->loadFromFile(filename, layerIndex)) {
        return Layer(std::move(layerImpl));
    }
    return Layer(); // Return empty layer on failure
}

Layer GpuLithoEngine::createLayerFromGeometry(const GeometryConfig& config) {
    auto layerImpl = std::make_unique<LayerImpl>();
    
    for (unsigned int gy = 0; gy < config.gridHeight; ++gy) {
        for (unsigned int gx = 0; gx < config.gridWidth; ++gx) {
            unsigned int centerX = config.centerX + gx * config.spacingX;
            unsigned int centerY = config.centerY + gy * config.spacingY;
            
            switch (config.shape) {
                case GeometryConfig::RECTANGLE:
                    layerImpl->generateRectangle(
                        centerX - config.width/2, 
                        centerY - config.height/2,
                        config.width, 
                        config.height);
                    break;
                    
                case GeometryConfig::CIRCLE:
                    layerImpl->generateCircle(centerX, centerY, config.radius, config.numSides);
                    break;
                    
                case GeometryConfig::REGULAR_POLYGON:
                    layerImpl->generateRegularPolygon(centerX, centerY, config.radius, config.numSides);
                    break;
                    
                case GeometryConfig::L_SHAPE:
                    layerImpl->generateLShape(centerX, centerY, config.width, config.height, config.thickness);
                    break;
            }
        }
    }
    
    return Layer(std::move(layerImpl));
}

Layer GpuLithoEngine::createLayerFromHostData(const uint2* h_vertices,
                                               const unsigned int* h_startIndices,
                                               const unsigned int* h_ptCounts,
                                               unsigned int polygonCount) {
    if (!h_vertices || !h_startIndices || !h_ptCounts || polygonCount == 0) {
        std::cerr << "Error: Invalid input data for createLayerFromHostData" << std::endl;
        return Layer();
    }

    // Calculate total vertex count from h_ptCounts
    unsigned int totalVertexCount = 0;
    for (unsigned int i = 0; i < polygonCount; ++i) {
        totalVertexCount += h_ptCounts[i];
    }

    if (totalVertexCount == 0) {
        std::cerr << "Error: Total vertex count is zero" << std::endl;
        return Layer();
    }

    auto layerImpl = std::make_unique<LayerImpl>(totalVertexCount, polygonCount);

    // Copy data to layer
    layerImpl->polygonCount = polygonCount;
    layerImpl->vertexCount = totalVertexCount;

    // Copy vertices
    std::memcpy(layerImpl->h_vertices, h_vertices, totalVertexCount * sizeof(uint2));

    // Copy start indices
    std::memcpy(layerImpl->h_startIndices, h_startIndices, polygonCount * sizeof(unsigned int));

    // Copy vertex counts
    std::memcpy(layerImpl->h_ptCounts, h_ptCounts, polygonCount * sizeof(unsigned int));

    return Layer(std::move(layerImpl));
}

Layer GpuLithoEngine::createLayerFromDeviceData(const uint2* d_vertices,
                                                 const unsigned int* d_startIndices,
                                                 const unsigned int* d_ptCounts,
                                                 unsigned int polygonCount) {
    if (!d_vertices || !d_startIndices || !d_ptCounts || polygonCount == 0) {
        std::cerr << "Error: Invalid input data for createLayerFromDeviceData" << std::endl;
        return Layer();
    }

    // First, copy d_ptCounts to host to calculate total vertex count
    std::vector<unsigned int> h_ptCounts_temp(polygonCount);
    CHECK_GPU_ERROR(gpuMemcpy(h_ptCounts_temp.data(), d_ptCounts,
                              polygonCount * sizeof(unsigned int),
                              gpuMemcpyDeviceToHost));

    // Calculate total vertex count
    unsigned int totalVertexCount = 0;
    for (unsigned int i = 0; i < polygonCount; ++i) {
        totalVertexCount += h_ptCounts_temp[i];
    }

    if (totalVertexCount == 0) {
        std::cerr << "Error: Total vertex count is zero" << std::endl;
        return Layer();
    }

    auto layerImpl = std::make_unique<LayerImpl>(totalVertexCount, polygonCount);

    // Set metadata
    layerImpl->polygonCount = polygonCount;
    layerImpl->vertexCount = totalVertexCount;

    // Copy device data to layer's device buffers
    layerImpl->allocateDevice();

    CHECK_GPU_ERROR(gpuMemcpy(layerImpl->d_vertices, d_vertices,
                              totalVertexCount * sizeof(uint2),
                              gpuMemcpyDeviceToDevice));

    CHECK_GPU_ERROR(gpuMemcpy(layerImpl->d_startIndices, d_startIndices,
                              polygonCount * sizeof(unsigned int),
                              gpuMemcpyDeviceToDevice));

    CHECK_GPU_ERROR(gpuMemcpy(layerImpl->d_ptCounts, d_ptCounts,
                              polygonCount * sizeof(unsigned int),
                              gpuMemcpyDeviceToDevice));

    // Copy to host as well
    layerImpl->copyToHost();

    return Layer(std::move(layerImpl));
}

Layer GpuLithoEngine::layerIntersection(const Layer& layer1, const Layer& layer2) {
    // Auto-prepare if not already done
    if (!impl->isGlobalBoxSet) {
        impl->prepareDualLayers(layer1, layer2);
    }
    return impl->performBooleanOperation(OperationType::INTERSECTION);
}

Layer GpuLithoEngine::layerUnion(const Layer& layer1, const Layer& layer2) {
    // Auto-prepare if not already done
    if (!impl->isGlobalBoxSet) {
        impl->prepareDualLayers(layer1, layer2);
    }
    return impl->performBooleanOperation(OperationType::UNION);
}

Layer GpuLithoEngine::layerDifference(const Layer& layer1, const Layer& layer2) {
    // Auto-prepare if not already done
    if (!impl->isGlobalBoxSet) {
        impl->prepareDualLayers(layer1, layer2);
    }
    return impl->performBooleanOperation(OperationType::DIFFERENCE);
}

Layer GpuLithoEngine::layerXor(const Layer& layer1, const Layer& layer2) {
    // XOR is implemented as: (layer1 DIFFERENCE layer2) + (layer2 DIFFERENCE layer1)
    // This ensures correct geometric computation using the DIFFERENCE algorithm twice

    // Compute layer1 - layer2
    if (!impl->isGlobalBoxSet) {
        impl->prepareDualLayers(layer1, layer2);
    }
    Layer diff1 = impl->performBooleanOperation(OperationType::DIFFERENCE);

    // Reset and compute layer2 - layer1
    impl->reset();
    impl->prepareDualLayers(layer2, layer1);  // Note: swapped order
    Layer diff2 = impl->performBooleanOperation(OperationType::DIFFERENCE);

    // Merge the two results into a single layer
    auto mergedLayer = std::make_unique<LayerImpl>();

    // Copy all polygons from diff1
    if (diff1.impl && !diff1.impl->empty()) {
        for (unsigned int i = 0; i < diff1.impl->polygonCount; ++i) {
            unsigned int start = diff1.impl->h_startIndices[i];
            unsigned int count = diff1.impl->h_ptCounts[i];
            mergedLayer->addPolygon(&diff1.impl->h_vertices[start], count);
        }
    }

    // Copy all polygons from diff2
    if (diff2.impl && !diff2.impl->empty()) {
        for (unsigned int i = 0; i < diff2.impl->polygonCount; ++i) {
            unsigned int start = diff2.impl->h_startIndices[i];
            unsigned int count = diff2.impl->h_ptCounts[i];
            mergedLayer->addPolygon(&diff2.impl->h_vertices[start], count);
        }
    }

    return Layer(std::move(mergedLayer));
}

Layer GpuLithoEngine::layerOffset(const Layer& layer, int offsetDistance) {
    // Auto-prepare if not already done
    if (!impl->isGlobalBoxSet) {
        impl->prepareSingleLayer(layer);
    }
    return impl->performSingleLayerOperation(OperationType::OFFSET, offsetDistance);
}

bool GpuLithoEngine::dumpLayerToFile(const Layer& layer, const std::string& filename, unsigned int layerIndex) {
    if (!layer.impl) {
        return false;
    }
    
    // If the engine has global box set, we need to restore original coordinates before saving
    if (impl->isGlobalBoxSet) {
        // Create a copy to avoid modifying the original layer
        LayerImpl layerCopy(*layer.impl);
        impl->restoreOriginalCoordinates(&layerCopy);
        return layerCopy.saveToFile(filename, layerIndex);
    } else {
        // No coordinate transformation needed
        return layer.impl->saveToFile(filename, layerIndex);
    }
}

bool GpuLithoEngine::visualizeLayer(const Layer& layer, const std::string& filename, bool showVertices) {
    if (!layer.impl || layer.impl->empty()) {
        return false;
    }
    
    // Create working copy for coordinate restoration if needed
    LayerImpl* workingLayer = layer.impl.get();
    std::unique_ptr<LayerImpl> layerCopy;
    
    if (impl->isGlobalBoxSet) {
        // Restore original coordinates for visualization
        layerCopy = std::make_unique<LayerImpl>(*layer.impl);
        impl->restoreOriginalCoordinates(layerCopy.get());
        workingLayer = layerCopy.get();
    }
    
    auto bbox = workingLayer->getBoundingBox();
    int width = bbox[2] - bbox[0] + 100;  // Add some padding
    int height = bbox[3] - bbox[1] + 100;
    
    // Create white background image explicitly
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Draw polygons
    for (unsigned int i = 0; i < workingLayer->polygonCount; ++i) {
        unsigned int start = workingLayer->h_startIndices[i];
        unsigned int count = workingLayer->h_ptCounts[i];
        
        std::vector<cv::Point> points;
        for (unsigned int j = 0; j < count; ++j) {
            int x = workingLayer->h_vertices[start + j].x - bbox[0] + 50;
            int y = workingLayer->h_vertices[start + j].y - bbox[1] + 50;
            points.emplace_back(x, y);
        }
        
        if (!points.empty()) {
            // Draw polygon outline in black on white background
            const cv::Point* pts = points.data();
            int npts = static_cast<int>(points.size());
            cv::polylines(image, &pts, &npts, 1, true, cv::Scalar(0, 0, 0), 1);
            
            // Optionally show vertices
            if (showVertices) {
                for (const auto& pt : points) {
                    cv::circle(image, pt, 2, cv::Scalar(0, 0, 255), -1);
                }
            }
        }
    }
    
    return cv::imwrite(filename, image);
}

void GpuLithoEngine::setMaxGridSize(unsigned int width, unsigned int height) {
    impl->maxGridWidth = width;
    impl->maxGridHeight = height;
}

std::vector<unsigned int> GpuLithoEngine::getCurrentGridSize() const {
    return {impl->currentGridWidth, impl->currentGridHeight};
}

std::vector<unsigned int> GpuLithoEngine::getGlobalBoundingBox() const {
    if (!impl->isGlobalBoxSet) {
        return {0, 0, 0, 0};
    }
    return {impl->globalMinX, impl->globalMinY, impl->globalMaxX, impl->globalMaxY};
}

bool GpuLithoEngine::isPrepared() const {
    return impl->isGlobalBoxSet;
}

void GpuLithoEngine::enableProfiling(bool enable) {
    impl->profilingEnabled = enable;
}

void GpuLithoEngine::printPerformanceStats() const {
    if (!impl->profilingEnabled || impl->operationCount == 0) {
        std::cout << "No performance data available (profiling disabled or no operations performed)\n";
        return;
    }
    
    std::cout << "=== GpuLithoLib Performance Statistics ===\n";
    std::cout << "Operations performed: " << impl->operationCount << "\n";
    std::cout << "Total ray casting time: " << impl->totalRayCastingTime << " ms\n";
    std::cout << "Total overlay time: " << impl->totalOverlayTime << " ms\n";
    std::cout << "Total contour extraction time: " << impl->totalContourTime << " ms\n";
    std::cout << "Average time per operation: " << 
        (impl->totalRayCastingTime + impl->totalOverlayTime + impl->totalContourTime) / impl->operationCount << " ms\n";
}

Layer GpuLithoEngine::createGroundTruthLayer(const std::string& filename, OperationType opType) {
    // Determine ground truth layer index based on operation type
    // Following the original main.cu mapping:
    unsigned int groundTruthLayerIndex;
    switch (opType) {
        case OperationType::INTERSECTION:
            groundTruthLayerIndex = 2;
            break;
        case OperationType::UNION:
            groundTruthLayerIndex = 3;
            break;
        case OperationType::DIFFERENCE:
            groundTruthLayerIndex = 4;
            break;
        case OperationType::XOR:
            groundTruthLayerIndex = 5;
            break;
        default:
            // No ground truth available for this operation
            std::cout << "No ground truth layer available for operation type\n";
            return Layer();
    }
    
    std::cout << "Loading ground truth layer " << groundTruthLayerIndex 
              << " for " << gpuLitho::OperationTypeUtils::operationTypeToString(opType) << " operation\n";
    
    return createLayerFromFile(filename, groundTruthLayerIndex);
}

bool GpuLithoEngine::visualizeVerificationComparison(const Layer& resultLayer, const Layer& groundTruthLayer, 
                                                    const std::string& filename) {
    if (!resultLayer.impl || resultLayer.impl->empty()) {
        std::cerr << "Error: Result layer is empty" << std::endl;
        return false;
    }
    
    if (!groundTruthLayer.impl || groundTruthLayer.impl->empty()) {
        std::cerr << "Error: Ground truth layer is empty" << std::endl;
        return false;
    }
    
    std::cout << "Creating verification comparison plot: " << filename << "\n";
    
    // Calculate combined bounding box for both layers
    auto resultBbox = resultLayer.impl->getBoundingBox();
    auto gtBbox = groundTruthLayer.impl->getBoundingBox();
    
    unsigned int minX = std::min(resultBbox[0], gtBbox[0]);
    unsigned int minY = std::min(resultBbox[1], gtBbox[1]);
    unsigned int maxX = std::max(resultBbox[2], gtBbox[2]);
    unsigned int maxY = std::max(resultBbox[3], gtBbox[3]);
    
    int width = maxX - minX + 100;  // Add padding
    int height = maxY - minY + 100;
    
    // Create white background image
    // Create white background image explicitly
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Draw ground truth polygons in green
    std::cout << "Drawing " << groundTruthLayer.impl->polygonCount << " ground truth polygons in green\n";
    for (unsigned int i = 0; i < groundTruthLayer.impl->polygonCount; ++i) {
        unsigned int start = groundTruthLayer.impl->h_startIndices[i];
        unsigned int count = groundTruthLayer.impl->h_ptCounts[i];
        
        std::vector<cv::Point> points;
        for (unsigned int j = 0; j < count; ++j) {
            int x = groundTruthLayer.impl->h_vertices[start + j].x - minX + 50;
            int y = groundTruthLayer.impl->h_vertices[start + j].y - minY + 50;
            points.emplace_back(x, y);
        }
        
        if (!points.empty()) {
            const cv::Point* pts = points.data();
            int npts = static_cast<int>(points.size());
            cv::polylines(image, &pts, &npts, 1, true, cv::Scalar(0, 255, 0), 2); // Green lines
        }
    }
    
    // Draw result polygons in red
    std::cout << "Drawing " << resultLayer.impl->polygonCount << " result polygons in red\n";
    for (unsigned int i = 0; i < resultLayer.impl->polygonCount; ++i) {
        unsigned int start = resultLayer.impl->h_startIndices[i];
        unsigned int count = resultLayer.impl->h_ptCounts[i];
        
        std::vector<cv::Point> points;
        for (unsigned int j = 0; j < count; ++j) {
            int x = resultLayer.impl->h_vertices[start + j].x - minX + 50;
            int y = resultLayer.impl->h_vertices[start + j].y - minY + 50;
            points.emplace_back(x, y);
        }
        
        if (!points.empty()) {
            const cv::Point* pts = points.data();
            int npts = static_cast<int>(points.size());
            cv::polylines(image, &pts, &npts, 1, true, cv::Scalar(0, 0, 255), 1); // Red lines
        }
    }
    
    // Add legend
    int legend_y = 30;
    cv::putText(image, "Green: Ground Truth", cv::Point(10, legend_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(image, "Red: Result", cv::Point(10, legend_y + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    
    // Add statistics
    cv::putText(image, "GT Polygons: " + std::to_string(groundTruthLayer.impl->polygonCount), 
                cv::Point(10, legend_y + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(image, "Result Polygons: " + std::to_string(resultLayer.impl->polygonCount), 
                cv::Point(10, legend_y + 70),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    return cv::imwrite(filename, image);
}

bool GpuLithoEngine::visualizeComprehensiveComparison(const Layer& subjectLayer, const Layer& clipperLayer,
                                                    const Layer& resultLayer, const Layer& groundTruthLayer,
                                                    const std::string& filename, bool useRawContours) {
    if (!subjectLayer.impl || !clipperLayer.impl || !resultLayer.impl || !groundTruthLayer.impl) {
        std::cerr << "Error: One or more layers are empty" << std::endl;
        return false;
    }
    
    std::cout << "Creating comprehensive comparison plot: " << filename << "\n";
    
    // Calculate combined bounding box for all layers
    auto subjectBbox = subjectLayer.impl->getBoundingBox();
    auto clipperBbox = clipperLayer.impl->getBoundingBox();
    auto resultBbox = resultLayer.impl->getBoundingBox();
    auto gtBbox = groundTruthLayer.impl->getBoundingBox();
    
    unsigned int minX = std::min({subjectBbox[0], clipperBbox[0], resultBbox[0], gtBbox[0]});
    unsigned int minY = std::min({subjectBbox[1], clipperBbox[1], resultBbox[1], gtBbox[1]});
    unsigned int maxX = std::max({subjectBbox[2], clipperBbox[2], resultBbox[2], gtBbox[2]});
    unsigned int maxY = std::max({subjectBbox[3], clipperBbox[3], resultBbox[3], gtBbox[3]});
    
    int width = maxX - minX + 100;
    int height = maxY - minY + 100;
    
    // Create white background image
    // Create white background image explicitly
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Helper lambda to draw polygons from a layer
    auto drawLayerPolygons = [&](const Layer& layer, cv::Scalar color, int thickness) {
        for (unsigned int i = 0; i < layer.impl->polygonCount; ++i) {
            unsigned int start = layer.impl->h_startIndices[i];
            unsigned int count = layer.impl->h_ptCounts[i];
            
            std::vector<cv::Point> points;
            for (unsigned int j = 0; j < count; ++j) {
                int x = layer.impl->h_vertices[start + j].x - minX + 50;
                int y = layer.impl->h_vertices[start + j].y - minY + 50;
                points.emplace_back(x, y);
            }
            
            if (!points.empty()) {
                const cv::Point* pts = points.data();
                int npts = static_cast<int>(points.size());
                cv::polylines(image, &pts, &npts, 1, true, color, thickness);
            }
        }
    };
    
    // Helper lambda to draw raw contours from a layer
    auto drawRawContours = [&](const Layer& layer, cv::Scalar color, int thickness) {
        // Convert layer to contours by extracting raw OpenCV contours
        // This simulates the detectRawContours method but for visualization
        auto tempLayer = std::make_unique<LayerImpl>(*layer.impl);
        tempLayer->ensureBitmapAllocated(width, height);
        tempLayer->clearBitmap();
        
        // Simple polygon-to-bitmap rendering
        for (unsigned int i = 0; i < layer.impl->polygonCount; ++i) {
            unsigned int start = layer.impl->h_startIndices[i];
            unsigned int count = layer.impl->h_ptCounts[i];
            
            std::vector<cv::Point> points;
            for (unsigned int j = 0; j < count; ++j) {
                int x = layer.impl->h_vertices[start + j].x - minX + 50;
                int y = layer.impl->h_vertices[start + j].y - minY + 50;
                points.emplace_back(x, y);
            }
            
            if (points.size() >= 3) {
                // Fill polygon in temporary bitmap (simplified)
                cv::Mat tempMat = cv::Mat::zeros(height, width, CV_8UC1);
                cv::fillPoly(tempMat, std::vector<std::vector<cv::Point>>{points}, cv::Scalar(255));
                
                // Find contours from filled polygon
                std::vector<std::vector<cv::Point>> contours;
                std::vector<cv::Vec4i> hierarchy;
                cv::findContours(tempMat, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                
                // Draw raw contours (with all points)
                for (const auto& contour : contours) {
                    if (contour.size() >= 2) {
                        for (size_t j = 0; j < contour.size(); ++j) {
                            size_t next_j = (j + 1) % contour.size();
                            cv::line(image, contour[j], contour[next_j], color, thickness);
                        }
                    }
                }
            }
        }
    };
    
    // Draw subject layer in magenta (clearly visible on any background)
    std::cout << "Drawing subject layer in magenta\n";
    drawLayerPolygons(subjectLayer, cv::Scalar(255, 0, 255), 1);
    
    // Draw clipper layer in orange  
    std::cout << "Drawing clipper layer in orange\n";
    drawLayerPolygons(clipperLayer, cv::Scalar(0, 165, 255), 1);
    
    // Draw ground truth in green
    std::cout << "Drawing ground truth in green\n";
    drawLayerPolygons(groundTruthLayer, cv::Scalar(0, 255, 0), 1);
    
    // Draw result layer in red (either raw contours or simplified polygons)
    if (useRawContours) {
        std::cout << "Drawing result raw contours in red\n";
        drawRawContours(resultLayer, cv::Scalar(0, 0, 255), 1);
    } else {
        std::cout << "Drawing result polygons in red\n";
        drawLayerPolygons(resultLayer, cv::Scalar(0, 0, 255), 1);
    }
    
    // Add comprehensive legend
    int legend_y = 30;
    cv::putText(image, "Magenta: Subject Layer", cv::Point(10, legend_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
    cv::putText(image, "Orange: Clipper Layer", cv::Point(10, legend_y + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 165, 255), 2);
    cv::putText(image, "Green: Ground Truth", cv::Point(10, legend_y + 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(image, useRawContours ? "Red: Raw Contours" : "Red: Result Polygons", 
                cv::Point(10, legend_y + 75),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    
    // Add statistics
    cv::putText(image, "Subject: " + std::to_string(subjectLayer.impl->polygonCount) + " polygons",
                cv::Point(10, legend_y + 105), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(image, "Clipper: " + std::to_string(clipperLayer.impl->polygonCount) + " polygons",
                cv::Point(10, legend_y + 125), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(image, "GT: " + std::to_string(groundTruthLayer.impl->polygonCount) + " polygons",
                cv::Point(10, legend_y + 145), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(image, "Result: " + std::to_string(resultLayer.impl->polygonCount) + " polygons",
                cv::Point(10, legend_y + 165), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    return cv::imwrite(filename, image);
}

} // namespace GpuLithoLib
