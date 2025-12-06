#include "../include/GpuLithoLib.h"
#include "LayerImpl.h"
#include "GpuOperations.cuh"
#include "IntersectionCompute.cuh"
#include "ContourProcessing.cuh"
#include "GpuKernelProfiler.cuh"
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

// Global profiler pointer (set by GpuLithoEngine)
GpuKernelProfiler* g_kernelProfiler = nullptr;

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

    // Intersection computation engine (persistent GPU memory)
    IntersectionComputeEngine intersectionEngine;

    // Contour detection and simplification engine
    ContourDetectEngine contourEngine;

    // GPU kernel profiler
    GpuKernelProfiler kernelProfiler;

    EngineImpl(unsigned int maxW, unsigned int maxH)
        : maxGridWidth(maxW), maxGridHeight(maxH),
          currentGridWidth(0), currentGridHeight(0),
          isGlobalBoxSet(false), globalMinX(0), globalMinY(0), globalMaxX(0), globalMaxY(0),
          shiftX(0), shiftY(0), profilingEnabled(false),
          totalRayCastingTime(0), totalOverlayTime(0), totalContourTime(0), operationCount(0) {
        // Set global profiler pointer
        g_kernelProfiler = &kernelProfiler;
    }

    ~EngineImpl() {
        // Clear global profiler pointer
        if (g_kernelProfiler == &kernelProfiler) {
            g_kernelProfiler = nullptr;
        }
    }
    
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
        
        // Add one pixel padding
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

        // Free intersection computation GPU memory
        intersectionEngine.freeData();
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
        
        // Launch ray casting kernel with timing
        gpuEvent_t rcStart, rcStop;
        gpuEventCreate(&rcStart);
        gpuEventCreate(&rcStop);
        gpuEventRecord(rcStart);

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

        gpuEventRecord(rcStop);
        gpuEventSynchronize(rcStop);

        float rcMs = 0.0f;
        gpuEventElapsedTime(&rcMs, rcStart, rcStop);
        kernelProfiler.addRayCastingTime(rcMs);

        gpuEventDestroy(rcStart);
        gpuEventDestroy(rcStop);
        
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

    // Perform optimized scanline ray casting on a layer (Range-based approach)
    void performScanlineRayCasting(LayerImpl* layer, int edgeMode = 1) {
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

        const unsigned int maxRangesPerScanline = 32;

        // Step 1: Allocate separate edge bitmap
        unsigned int bitmapSize = currentGridWidth * currentGridHeight;
        unsigned int* d_edgeBitmap = nullptr;
        CHECK_GPU_ERROR(gpuMalloc(&d_edgeBitmap, bitmapSize * sizeof(unsigned int)));
        CHECK_GPU_ERROR(gpuMemset(d_edgeBitmap, 0, bitmapSize * sizeof(unsigned int)));

        // Step 2: Calculate total scanlines needed and scanline offsets
        // Copy bounding boxes to host to compute offsets
        thrust::device_vector<uint4> d_boxes_vec(layer->d_boxes, layer->d_boxes + layer->polygonCount);
        thrust::host_vector<uint4> h_boxes = d_boxes_vec;

        thrust::host_vector<unsigned int> h_scanlineOffsets(layer->polygonCount + 1);
        unsigned int totalScanlines = 0;
        for (unsigned int i = 0; i < layer->polygonCount; i++) {
            h_scanlineOffsets[i] = totalScanlines;
            unsigned int boxHeight = h_boxes[i].w - h_boxes[i].y;
            totalScanlines += boxHeight;
        }
        h_scanlineOffsets[layer->polygonCount] = totalScanlines;

        // Allocate device buffers for scanline ranges
        unsigned int* d_scanlineOffsets = nullptr;
        uint2* d_scanlineRanges = nullptr;
        unsigned int* d_scanlineRangeCounts = nullptr;

        CHECK_GPU_ERROR(gpuMalloc(&d_scanlineOffsets, (layer->polygonCount + 1) * sizeof(unsigned int)));
        CHECK_GPU_ERROR(gpuMalloc(&d_scanlineRanges, totalScanlines * maxRangesPerScanline * sizeof(uint2)));
        CHECK_GPU_ERROR(gpuMalloc(&d_scanlineRangeCounts, totalScanlines * sizeof(unsigned int)));

        CHECK_GPU_ERROR(gpuMemcpy(d_scanlineOffsets, h_scanlineOffsets.data(),
                                  (layer->polygonCount + 1) * sizeof(unsigned int), gpuMemcpyHostToDevice));
        CHECK_GPU_ERROR(gpuMemset(d_scanlineRangeCounts, 0, totalScanlines * sizeof(unsigned int)));

        // Step 3: Render edges to edge bitmap
        gpuEvent_t rcStart, rcStop;
        gpuEventCreate(&rcStart);
        gpuEventCreate(&rcStop);
        gpuEventRecord(rcStart);

        edgeRender_kernel<<<layer->polygonCount, 512>>>(
            layer->d_vertices,
            layer->d_startIndices,
            layer->d_ptCounts,
            d_edgeBitmap,
            currentGridWidth,
            currentGridHeight,
            1);  // mode=1 to render edges
        CHECK_GPU_ERROR(gpuGetLastError());

        // Step 4: Check edge-right neighbors and mark inside/outside
        checkEdgeRightNeighbor_kernel<<<layer->polygonCount, 512>>>(
            layer->d_vertices,
            layer->d_startIndices,
            layer->d_ptCounts,
            layer->d_boxes,
            d_edgeBitmap,
            currentGridWidth,
            currentGridHeight,
            layer->polygonCount);
        CHECK_GPU_ERROR(gpuGetLastError());

        // Step 5: Find scanline ranges
        findScanlineRanges_kernel<<<layer->polygonCount, 512>>>(
            d_edgeBitmap,
            layer->d_boxes,
            d_scanlineOffsets,
            d_scanlineRanges,
            d_scanlineRangeCounts,
            currentGridWidth,
            currentGridHeight,
            layer->polygonCount,
            maxRangesPerScanline);
        CHECK_GPU_ERROR(gpuGetLastError());

        // Step 6: Render polygon interiors using ranges
        renderScanlineRanges_kernel<<<layer->polygonCount, 512>>>(
            layer->d_boxes,
            d_scanlineOffsets,
            d_scanlineRanges,
            d_scanlineRangeCounts,
            layer->d_bitmap,
            currentGridWidth,
            currentGridHeight,
            layer->polygonCount,
            maxRangesPerScanline);
        CHECK_GPU_ERROR(gpuGetLastError());

        gpuEventRecord(rcStop);
        gpuEventSynchronize(rcStop);

        float rcMs = 0.0f;
        gpuEventElapsedTime(&rcMs, rcStart, rcStop);
        kernelProfiler.addRayCastingTime(rcMs);

        gpuEventDestroy(rcStart);
        gpuEventDestroy(rcStop);

        // Debug: Save bitmap visualization with polygon outlines
        {
            static int debugCounter = 0;
            std::string filename = "debug_scanline_bitmap_" + std::to_string(debugCounter++) + ".png";

            // Copy bitmap from device to host
            std::vector<unsigned int> h_bitmap(bitmapSize);
            CHECK_GPU_ERROR(gpuMemcpy(h_bitmap.data(), layer->d_bitmap,
                                      bitmapSize * sizeof(unsigned int), gpuMemcpyDeviceToHost));

            // Create color image for visualization
            cv::Mat image(currentGridHeight, currentGridWidth, CV_8UC3, cv::Scalar(255, 255, 255));

            // Draw filled regions (blue for any polygon)
            for (unsigned int y = 0; y < currentGridHeight; y++) {
                for (unsigned int x = 0; x < currentGridWidth; x++) {
                    unsigned int val = h_bitmap[y * currentGridWidth + x];
                    if (val > 0) {
                        image.at<cv::Vec3b>(y, x) = cv::Vec3b(200, 150, 100);  // Light blue fill
                    }
                }
            }

            // Draw polygon outlines using CPU vertex data
            for (unsigned int p = 0; p < layer->polygonCount; p++) {
                unsigned int startIdx = layer->h_startIndices[p];
                unsigned int ptCount = layer->h_ptCounts[p];

                for (unsigned int i = 0; i < ptCount; i++) {
                    uint2 v1 = layer->h_vertices[startIdx + i];
                    uint2 v2 = layer->h_vertices[startIdx + (i + 1) % ptCount];
                        // define a color based on polygon index
                        cv::Scalar color;
                        if (p <= 102) {
                            color = cv::Scalar(0, 255, 0);  // Green
                        } else if (p <= 103) {
                            color = cv::Scalar(0, 0, 255);  // Red
                        } else if (p <= 105) {
                            color = cv::Scalar(255, 0, 0);  // Blue
                        }
                        else {
                            color = cv::Scalar(0, 0, 0);    // Black
                        }

                        cv::line(image,
                                cv::Point(v1.x, v1.y),
                                cv::Point(v2.x, v2.y),
                                color,
                                1, cv::LINE_AA);
                }
            }

            cv::imwrite(filename, image);
            std::cout << "Saved debug bitmap: " << filename << std::endl;
        }

        // Clean up temporary buffers
        gpuFree(d_edgeBitmap);
        gpuFree(d_scanlineOffsets);
        gpuFree(d_scanlineRanges);
        gpuFree(d_scanlineRangeCounts);

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

        // Time the overlay kernel
        gpuEvent_t ovStart, ovStop;
        gpuEventCreate(&ovStart);
        gpuEventCreate(&ovStop);
        gpuEventRecord(ovStart);

        overlay_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            subject->d_bitmap,
            clipper->d_bitmap,
            output->d_bitmap,
            currentGridWidth,
            currentGridHeight);

        CHECK_GPU_ERROR(gpuGetLastError());
        gpuEventRecord(ovStop);
        gpuEventSynchronize(ovStop);

        float ovMs = 0.0f;
        gpuEventElapsedTime(&ovMs, ovStart, ovStop);
        kernelProfiler.addOverlayTime(ovMs);

        gpuEventDestroy(ovStart);
        gpuEventDestroy(ovStop);
        
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

    // Perform boolean operation with full pipeline
    Layer performBooleanOperation(OperationType opType) {
        if (!preparedLayer1 || !preparedLayer2) {
            std::cerr << "Error: Layers not prepared for boolean operation" << std::endl;
            return Layer();
        }

        // Reset GPU kernel timers at the start of boolean operation
        kernelProfiler.reset();

        auto output = std::make_unique<LayerImpl>();

        // ========== DEBUG: Visualize preparedLayer2 (clipper) using CPU vertex data ==========
        {
            std::cout << "\n========== DEBUG: Visualizing preparedLayer2 (clipper layer) ==========" << std::endl;

            // Create image with white background
            cv::Mat clipperImage(currentGridHeight, currentGridWidth, CV_8UC3, cv::Scalar(255, 255, 255));

            // Draw 200x200 grid
            cv::Scalar gridColor(200, 200, 200);  // Light gray
            for (int x = 0; x < currentGridWidth; x += 200) {
                cv::line(clipperImage, cv::Point(x, 0), cv::Point(x, currentGridHeight - 1), gridColor, 1);
            }
            for (int y = 0; y < currentGridHeight; y += 200) {
                cv::line(clipperImage, cv::Point(0, y), cv::Point(currentGridWidth - 1, y), gridColor, 1);
            }

            // Draw polygons using CPU vertex data
            unsigned int numPolygons = preparedLayer2->polygonCount;
            std::cout << "  Number of polygons in preparedLayer2: " << numPolygons << std::endl;

            for (unsigned int polyIdx = 0; polyIdx < numPolygons; ++polyIdx) {
                unsigned int startIdx = preparedLayer2->h_startIndices[polyIdx];
                unsigned int ptCount = preparedLayer2->h_ptCounts[polyIdx];

                if (ptCount < 3) continue;

                // Draw polygon edges in red
                for (unsigned int i = 0; i < ptCount; ++i) {
                    unsigned int nextI = (i + 1) % ptCount;
                    uint2 p1 = preparedLayer2->h_vertices[startIdx + i];
                    uint2 p2 = preparedLayer2->h_vertices[startIdx + nextI];
                    cv::line(clipperImage, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), cv::Scalar(0, 0, 255), 1);
                }

                // Draw polygon ID label near first vertex
                uint2 firstVert = preparedLayer2->h_vertices[startIdx];
                std::string label = std::to_string(polyIdx);
                cv::putText(clipperImage, label, cv::Point(firstVert.x + 2, firstVert.y - 2),
                            cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 0), 1);
            }

            // Draw tick marks and labels on axes
            cv::Scalar tickColor(0, 0, 0);  // Black
            for (int x = 0; x < currentGridWidth; x += 200) {
                cv::line(clipperImage, cv::Point(x, 0), cv::Point(x, 10), tickColor, 2);
                cv::putText(clipperImage, std::to_string(x), cv::Point(x + 2, 25),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, tickColor, 1);
            }
            for (int y = 0; y < currentGridHeight; y += 200) {
                cv::line(clipperImage, cv::Point(0, y), cv::Point(10, y), tickColor, 2);
                cv::putText(clipperImage, std::to_string(y), cv::Point(12, y + 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, tickColor, 1);
            }

            cv::imwrite("debug_clipper_layer_vertices.png", clipperImage);
            std::cout << "  Saved to debug_clipper_layer_vertices.png" << std::endl;
            std::cout << "========== END DEBUG: clipper layer visualization ==========" << std::endl;
        }

        // Step 1: Ray casting for both layers
        // Old method (commented out):
        // performRayCasting(preparedLayer1.get(), (opType == OperationType::DIFFERENCE) ? 1 : 1);
        // performRayCasting(preparedLayer2.get(), (opType == OperationType::DIFFERENCE) ? 0 : 1);

        // New optimized scanline method:
        cudaDeviceSynchronize();  // Ensure first layer is done before second
        fflush(0);
        printf("Starting scanline ray casting for layer 1...\n");
        performScanlineRayCasting(preparedLayer1.get(), (opType == OperationType::DIFFERENCE) ? 1 : 1);
        cudaDeviceSynchronize();  // Ensure first layer is done before second
        printf("Starting scanline ray casting for layer 2...\n");
        fflush(0);
        performScanlineRayCasting(preparedLayer2.get(), (opType == OperationType::DIFFERENCE) ? 0 : 1);
        cudaDeviceSynchronize();  // Ensure first layer is done before second
        fflush(0);

        // ========== DEBUG: Check preparedLayer2 bitmap at index 2247600 ==========
        {
            unsigned int debug_idx = 2247600;
            unsigned int h_clipper_pixel_val = 0;
            CHECK_GPU_ERROR(gpuMemcpy(&h_clipper_pixel_val, preparedLayer2->d_bitmap + debug_idx,
                                       sizeof(unsigned int), gpuMemcpyDeviceToHost));
            std::cout << "\n========== DEBUG: preparedLayer2 bitmap after scanline raycasting ==========" << std::endl;
            std::cout << "  preparedLayer2->d_bitmap[" << debug_idx << "] = " << h_clipper_pixel_val << std::endl;
            std::cout << "========== END DEBUG ==========" << std::endl;
        }

        // Step 2: Overlay
        performOverlay(preparedLayer1.get(), preparedLayer2.get(), output.get());

        // ========== DEBUG: Check output bitmap at index 2247600 after overlay ==========
        {
            unsigned int debug_idx = 2247600;
            unsigned int h_output_pixel_val = 0;
            CHECK_GPU_ERROR(gpuMemcpy(&h_output_pixel_val, output->d_bitmap + debug_idx,
                                       sizeof(unsigned int), gpuMemcpyDeviceToHost));
            unsigned int subject_part = h_output_pixel_val & 0xFFFF;
            unsigned int clipper_part = (h_output_pixel_val >> 16) & 0xFFFF;
            std::cout << "\n========== DEBUG: output bitmap after overlay ==========" << std::endl;
            std::cout << "  output->d_bitmap[" << debug_idx << "] = " << h_output_pixel_val
                      << " (subject=" << subject_part << ", clipper=" << clipper_part << ")" << std::endl;
            std::cout << "========== END DEBUG ==========" << std::endl;
        }

        // Step 3: Compute intersection points from overlay result
        auto intersection_points_set = intersectionEngine.computeAllIntersectionPoints(
            output.get(), preparedLayer1.get(), preparedLayer2.get(),
            currentGridWidth, currentGridHeight);
        
        // Step 4: Detect raw contours
        auto raw_contours = contourEngine.detectRawContours(output.get(), opType, currentGridWidth, currentGridHeight);

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
        auto simplified_contours = contourEngine.simplifyContoursWithGeometry(
            raw_contours, preparedLayer1.get(), preparedLayer2.get(),
            output.get(), intersection_points_set, opType,
            currentGridWidth, currentGridHeight);
        
        // Step 6: Convert simplified contours to Layer
        auto resultLayer = std::make_unique<LayerImpl>();
        for (const auto& contour : simplified_contours) {
            if (contour.size() >= 3) {
                std::vector<uint2> vertices;
                for (const auto& pt : contour) {
                    uint2 v;
                    v.x = static_cast<unsigned int>(pt.x);
                    v.y = static_cast<unsigned int>(pt.y);
                    vertices.push_back(v);
                }
                resultLayer->addPolygon(vertices.data(), vertices.size());
            }
        }
        
        // Step 7: Restore original coordinates
        restoreOriginalCoordinates(resultLayer.get());

        // Print GPU kernel timing summary
        kernelProfiler.printSummary();

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
            // Old method (commented out):
            // performRayCasting(offsetLayer.get(), 1);

            // New optimized scanline method:
            performScanlineRayCasting(offsetLayer.get(), 1);
            
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
