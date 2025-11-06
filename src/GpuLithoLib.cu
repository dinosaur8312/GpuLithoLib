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

namespace GpuLithoLib {

using gpuLitho::OperationType;

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
        
        // Step 1: Extract intersecting polygon pairs from overlay bitmap
        std::set<std::pair<unsigned int, unsigned int>> intersecting_pairs;
        
        // Ensure bitmap data is on host
        outputLayer->copyBitmapToHost();
        
        // Scan through bitmap to find intersecting pairs
        for (unsigned int y = 0; y < currentGridHeight; ++y) {
            for (unsigned int x = 0; x < currentGridWidth; ++x) {
                unsigned int pixel_value = outputLayer->h_bitmap[y * currentGridWidth + x];
                
                // Extract subject and clipper polygon IDs
                unsigned int subject_id = pixel_value & 0xFFFF;        // Lower 16 bits
                unsigned int clipper_id = (pixel_value >> 16) & 0xFFFF; // Upper 16 bits
                
                // Check if this pixel represents an intersection (both IDs non-zero)
                if (subject_id > 0 && clipper_id > 0) {
                    intersecting_pairs.insert(std::make_pair(subject_id, clipper_id));
                }
            }
        }
        
        // Step 2: Compute intersection points for each pair
        // Note: This is a simplified version. The full implementation would need
        // edge-edge intersection computation similar to bitmap_layer.cu's computeIntersectionPoints
        for (const auto& pair : intersecting_pairs) {
            unsigned int subject_id = pair.first - 1;  // Convert from 1-based to 0-based
            unsigned int clipper_id = pair.second - 1;
            
            if (subject_id >= subjectLayer->polygonCount || clipper_id >= clipperLayer->polygonCount) {
                continue;
            }
            
            std::set<IntersectionPoint> pair_intersections;
            
            // Add subject vertices as candidate points
            unsigned int subject_start = subjectLayer->h_startIndices[subject_id];
            unsigned int subject_count = subjectLayer->h_ptCounts[subject_id];
            for (unsigned int i = 0; i < subject_count; ++i) {
                unsigned int x = subjectLayer->h_vertices[subject_start + i].x;
                unsigned int y = subjectLayer->h_vertices[subject_start + i].y;
                pair_intersections.emplace(x, y, 1.5f, PointType::SUBJECT_VERTEX);
            }
            
            // Add clipper vertices as candidate points
            unsigned int clipper_start = clipperLayer->h_startIndices[clipper_id];
            unsigned int clipper_count = clipperLayer->h_ptCounts[clipper_id];
            for (unsigned int i = 0; i < clipper_count; ++i) {
                unsigned int x = clipperLayer->h_vertices[clipper_start + i].x;
                unsigned int y = clipperLayer->h_vertices[clipper_start + i].y;
                pair_intersections.emplace(x, y, 1.5f, PointType::CLIPPER_VERTEX);
            }
            
            // TODO: Add actual edge-edge intersection computation here
            // For now, we only include polygon vertices
            
            intersection_points_set[pair] = std::move(pair_intersections);
        }
        
        return intersection_points_set;
    }
    
    // Detect raw contours from output layer bitmap
    std::vector<std::vector<cv::Point>> detectRawContours(LayerImpl* outputLayer, OperationType opType) {
        std::vector<std::vector<cv::Point>> contours;
        
        if (!outputLayer) {
            return contours;
        }
        
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
        
        // Use OpenCV to find contours
        cv::Mat binaryImage(currentGridHeight, currentGridWidth, CV_8UC1);
        for (int y = 0; y < currentGridHeight; ++y) {
            for (int x = 0; x < currentGridWidth; ++x) {
                int idx = y * currentGridWidth + x;
                binaryImage.at<uchar>(y, x) = contourLayer->h_bitmap[idx] > 0 ? 255 : 0;
            }
        }
        
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binaryImage, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
        
        return contours;
    }
    
    // Simplify contours using layer vertex information and intersection points
    // Input: raw contours, two layers, intersection points set
    // Output: geometrically accurate simplified contours
    std::vector<std::vector<cv::Point>> simplifyContoursWithGeometry(
        const std::vector<std::vector<cv::Point>>& raw_contours,
        LayerImpl* subjectLayer,
        LayerImpl* clipperLayer,
        LayerImpl* outputLayer,
        const std::map<std::pair<unsigned int, unsigned int>, std::set<IntersectionPoint>>& intersection_points_set) {
        
        std::vector<std::vector<cv::Point>> simplified_contours;
        
        if (!outputLayer || !subjectLayer || !clipperLayer) {
            return simplified_contours;
        }
        
        // Ensure bitmap is on host
        outputLayer->copyBitmapToHost();
        
        // Process each contour
        for (const auto& contour : raw_contours) {
            if (contour.size() < 3) {
                continue;
            }
            
            // Determine which polygons this contour belongs to
            std::set<unsigned int> subject_ids;
            std::set<unsigned int> clipper_ids;
            
            // Sample points to determine polygon IDs
            for (const cv::Point& pt : contour) {
                if (pt.x >= 0 && pt.x < currentGridWidth && pt.y >= 0 && pt.y < currentGridHeight) {
                    unsigned int pixel_idx = pt.y * currentGridWidth + pt.x;
                    unsigned int pixel_value = outputLayer->h_bitmap[pixel_idx];
                    
                    if (pixel_value > 0) {
                        unsigned int clipper_id = (pixel_value >> 16) & 0xFFFF;
                        unsigned int subject_id = pixel_value & 0xFFFF;
                        
                        if (subject_id > 0) subject_ids.insert(subject_id);
                        if (clipper_id > 0) clipper_ids.insert(clipper_id);
                    }
                }
            }
            
            // Handle pure subject or pure clipper cases
            if (subject_ids.empty() && !clipper_ids.empty()) {
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
            else if (!subject_ids.empty() && !clipper_ids.empty()) {
                // Intersection case - use intersection points to guide simplification
                std::vector<cv::Point> simplified_contour;
                
                // Collect all candidate points from intersection_points_set
                std::vector<std::pair<unsigned int, unsigned int>> candidate_points;
                
                for (unsigned int subject_id : subject_ids) {
                    for (unsigned int clipper_id : clipper_ids) {
                        auto pair_key = std::make_pair(subject_id, clipper_id);
                        auto it = intersection_points_set.find(pair_key);
                        if (it != intersection_points_set.end()) {
                            for (const auto& pt : it->second) {
                                candidate_points.push_back(pt.position);
                            }
                        }
                    }
                }
                
                // Match contour points to candidate points
                for (const cv::Point& contour_pt : contour) {
                    // Find closest candidate point within threshold
                    bool matched = false;
                    for (const auto& candidate : candidate_points) {
                        float dx = contour_pt.x - candidate.first;
                        float dy = contour_pt.y - candidate.second;
                        float dist = std::sqrt(dx*dx + dy*dy);
                        
                        if (dist <= 2.0f) {  // Use threshold
                            simplified_contour.emplace_back(candidate.first, candidate.second);
                            matched = true;
                            break;
                        }
                    }
                    
                    // If no candidate matched, check if it's a corner point in raw contour
                    if (!matched) {
                        // Simple corner detection: check angle change
                        // For now, keep every Nth point as simplification
                        static int skip_counter = 0;
                        if (skip_counter++ % 5 == 0) {
                            simplified_contour.push_back(contour_pt);
                        }
                    }
                }
                
                if (simplified_contour.size() > 2) {
                    simplified_contours.push_back(std::move(simplified_contour));
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
        
        // Step 5: Simplify contours using geometry
        auto simplified_contours = simplifyContoursWithGeometry(
            raw_contours, preparedLayer1.get(), preparedLayer2.get(), 
            output.get(), intersection_points_set);
        
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
    // Auto-prepare if not already done
    if (!impl->isGlobalBoxSet) {
        impl->prepareDualLayers(layer1, layer2);
    }
    return impl->performBooleanOperation(OperationType::XOR);
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