#pragma once

#include <string>
#include <vector>
#include <memory>
#include "gpuLitho.h"
#include "operation_types.h"

namespace GpuLithoLib {

// Forward declarations
class LayerImpl;

// Import OperationType from gpuLitho namespace
using gpuLitho::OperationType;

/**
 * @brief Handle to a polygon layer in GPU memory
 * 
 * This class represents a collection of polygons that can be processed
 * efficiently on the GPU. All operations are performed on GPU memory
 * with automatic memory management.
 */
class Layer {
public:
    /**
     * @brief Default constructor - creates empty layer
     */
    Layer();
    
    /**
     * @brief Copy constructor - performs deep copy
     */
    Layer(const Layer& other);
    
    /**
     * @brief Assignment operator - performs deep copy
     */
    Layer& operator=(const Layer& other);
    
    /**
     * @brief Move constructor
     */
    Layer(Layer&& other) noexcept;
    
    /**
     * @brief Move assignment operator
     */
    Layer& operator=(Layer&& other) noexcept;
    
    /**
     * @brief Destructor - automatically cleans up GPU memory
     */
    ~Layer();
    
    /**
     * @brief Check if layer is empty
     * @return true if layer contains no polygons
     */
    bool empty() const;
    
    /**
     * @brief Get number of polygons in layer
     * @return polygon count
     */
    unsigned int getPolygonCount() const;
    
    /**
     * @brief Get total number of vertices in layer
     * @return vertex count
     */
    unsigned int getVertexCount() const;
    
    /**
     * @brief Get bounding box of all polygons in layer
     * @return {minX, minY, maxX, maxY}
     */
    std::vector<unsigned int> getBoundingBox() const;

    /**
     * @brief Get host vertex buffer pointer (read-only)
     * @return Pointer to host vertex array (uint2*), or nullptr if layer is empty
     */
    const uint2* getHostVertices() const;

    /**
     * @brief Get host start indices buffer pointer (read-only)
     * @return Pointer to host start indices array, or nullptr if layer is empty
     */
    const unsigned int* getHostStartIndices() const;

    /**
     * @brief Get host vertex counts buffer pointer (read-only)
     * @return Pointer to host vertex counts array, or nullptr if layer is empty
     */
    const unsigned int* getHostPtCounts() const;

    /**
     * @brief Get device vertex buffer pointer (read-only)
     * @return Pointer to device vertex array (uint2*), or nullptr if layer is empty
     */
    const uint2* getDeviceVertices() const;

    /**
     * @brief Get device start indices buffer pointer (read-only)
     * @return Pointer to device start indices array, or nullptr if layer is empty
     */
    const unsigned int* getDeviceStartIndices() const;

    /**
     * @brief Get device vertex counts buffer pointer (read-only)
     * @return Pointer to device vertex counts array, or nullptr if layer is empty
     */
    const unsigned int* getDevicePtCounts() const;

private:
    friend class GpuLithoEngine;
    std::unique_ptr<LayerImpl> impl;
    
    // Private constructor for internal use
    explicit Layer(std::unique_ptr<LayerImpl> impl);
};

/**
 * @brief Configuration for geometric layer creation
 */
struct GeometryConfig {
    enum ShapeType {
        RECTANGLE,
        CIRCLE,
        REGULAR_POLYGON,
        L_SHAPE
    };
    
    ShapeType shape = RECTANGLE;
    
    // Position and size
    unsigned int centerX = 0;
    unsigned int centerY = 0;
    unsigned int width = 100;
    unsigned int height = 100;
    
    // For circles and regular polygons
    unsigned int radius = 50;
    unsigned int numSides = 6;  // For regular polygons
    
    // For L-shapes
    unsigned int thickness = 20;
    
    // Grid parameters (for creating multiple shapes)
    unsigned int gridWidth = 1;
    unsigned int gridHeight = 1;
    unsigned int spacingX = 200;
    unsigned int spacingY = 200;
};

/**
 * @brief Main GPU Lithography Engine
 * 
 * This class provides a simple API for performing lithography operations
 * on polygon layers using GPU acceleration. The engine maintains coordinate
 * system information and handles layer preparation automatically.
 */
class GpuLithoEngine {
public:
    /**
     * @brief Constructor
     * @param maxGridWidth Maximum working area width (default: 10000)
     * @param maxGridHeight Maximum working area height (default: 10000)
     */
    explicit GpuLithoEngine(unsigned int maxGridWidth = 10000, unsigned int maxGridHeight = 10000);
    
    /**
     * @brief Destructor
     */
    ~GpuLithoEngine();
    
    // === Layer Creation Methods ===
    
    /**
     * @brief Create layer from file
     * @param filename Path to polygon file
     * @param layerIndex Layer index to read (default: 0)
     * @return Layer handle
     */
    Layer createLayerFromFile(const std::string& filename, unsigned int layerIndex = 0);
    
    /**
     * @brief Create layer from geometric configuration
     * @param config Geometry configuration
     * @return Layer handle
     */
    Layer createLayerFromGeometry(const GeometryConfig& config);

    /**
     * @brief Create layer from host vertex data
     * @param h_vertices Pointer to vertex array on host (uint2 array)
     * @param h_startIndices Pointer to start indices array on host
     * @param h_ptCounts Pointer to vertex counts array on host
     * @param polygonCount Number of polygons
     * @return Layer handle (vertexCount calculated internally from h_ptCounts)
     */
    Layer createLayerFromHostData(const uint2* h_vertices,
                                   const unsigned int* h_startIndices,
                                   const unsigned int* h_ptCounts,
                                   unsigned int polygonCount);

    /**
     * @brief Create layer from device vertex data
     * @param d_vertices Pointer to vertex array on device (uint2 array)
     * @param d_startIndices Pointer to start indices array on device
     * @param d_ptCounts Pointer to vertex counts array on device
     * @param polygonCount Number of polygons
     * @return Layer handle (vertexCount calculated internally from d_ptCounts)
     */
    Layer createLayerFromDeviceData(const uint2* d_vertices,
                                     const unsigned int* d_startIndices,
                                     const unsigned int* d_ptCounts,
                                     unsigned int polygonCount);

    // === Layer Preparation ===
    
    /**
     * @brief Prepare engine for single layer operation
     * Calculates global bounding box and sets up coordinate system
     * @param layer Input layer
     */
    void prepareSingleLayer(const Layer& layer);
    
    /**
     * @brief Prepare engine for dual layer operation  
     * Calculates combined global bounding box and sets up coordinate system
     * @param layer1 First input layer
     * @param layer2 Second input layer
     */
    void prepareDualLayers(const Layer& layer1, const Layer& layer2);
    
    /**
     * @brief Reset engine state (clears global bounding box and prepared layers)
     */
    void reset();
    
    // === Boolean Operations ===
    
    /**
     * @brief Compute intersection of two layers
     * Automatically calls prepareDualLayers if not already prepared
     * @param layer1 First input layer
     * @param layer2 Second input layer
     * @return Result layer containing intersection
     */
    Layer layerIntersection(const Layer& layer1, const Layer& layer2);
    
    /**
     * @brief Compute union of two layers
     * Automatically calls prepareDualLayers if not already prepared
     * @param layer1 First input layer
     * @param layer2 Second input layer
     * @return Result layer containing union
     */
    Layer layerUnion(const Layer& layer1, const Layer& layer2);
    
    /**
     * @brief Compute difference of two layers (layer1 - layer2)
     * Automatically calls prepareDualLayers if not already prepared
     * @param layer1 Subject layer
     * @param layer2 Clipper layer
     * @return Result layer containing difference
     */
    Layer layerDifference(const Layer& layer1, const Layer& layer2);
    
    /**
     * @brief Compute XOR of two layers
     * Automatically calls prepareDualLayers if not already prepared
     * @param layer1 First input layer
     * @param layer2 Second input layer
     * @return Result layer containing XOR
     */
    Layer layerXor(const Layer& layer1, const Layer& layer2);
    
    // === Geometric Operations ===
    
    /**
     * @brief Offset (grow/shrink) layer
     * Automatically calls prepareSingleLayer if not already prepared
     * @param layer Input layer
     * @param offsetDistance Offset distance in pixels (positive = grow, negative = shrink)
     * @return Result layer containing offset polygons
     */
    Layer layerOffset(const Layer& layer, int offsetDistance);
    
    // === Output Methods ===
    
    /**
     * @brief Export layer to file
     * @param layer Layer to export
     * @param filename Output filename
     * @param layerIndex Target layer index in file (default: 0)
     * @return true if successful
     */
    bool dumpLayerToFile(const Layer& layer, const std::string& filename, unsigned int layerIndex = 0);
    
    /**
     * @brief Create visualization image of layer
     * @param layer Layer to visualize
     * @param filename Output image filename (.png)
     * @param showVertices Whether to highlight vertices (default: false)
     * @return true if successful
     */
    bool visualizeLayer(const Layer& layer, const std::string& filename, bool showVertices = false);
    
    // === Configuration ===
    
    /**
     * @brief Set maximum working area size
     * @param width Maximum grid width
     * @param height Maximum grid height
     */
    void setMaxGridSize(unsigned int width, unsigned int height);
    
    /**
     * @brief Get current actual grid size (after layer preparation)
     * @return {width, height}
     */
    std::vector<unsigned int> getCurrentGridSize() const;
    
    /**
     * @brief Get global bounding box information
     * @return {minX, minY, maxX, maxY} in original coordinates, or {0,0,0,0} if not prepared
     */
    std::vector<unsigned int> getGlobalBoundingBox() const;
    
    /**
     * @brief Check if engine is prepared for operations
     * @return true if global bounding box is set
     */
    bool isPrepared() const;
    
    /**
     * @brief Enable/disable performance profiling
     * @param enable Enable profiling
     */
    void enableProfiling(bool enable);
    
    /**
     * @brief Print performance statistics
     */
    void printPerformanceStats() const;
    
    // === Ground Truth Comparison ===
    
    /**
     * @brief Create ground truth layer for the specified operation type
     * @param filename Path to polygon file containing ground truth data
     * @param opType Operation type to determine which layer index to load
     * @return Layer handle for ground truth, or empty layer if not available
     */
    Layer createGroundTruthLayer(const std::string& filename, OperationType opType);
    
    /**
     * @brief Create verification comparison visualization
     * @param resultLayer Result layer from boolean operation
     * @param groundTruthLayer Expected ground truth layer
     * @param filename Output image filename
     * @return true if successful
     */
    bool visualizeVerificationComparison(const Layer& resultLayer, const Layer& groundTruthLayer, 
                                       const std::string& filename);
    
    /**
     * @brief Create comprehensive comparison visualization showing all layers
     * @param subjectLayer Subject input layer
     * @param clipperLayer Clipper input layer
     * @param resultLayer Result layer from boolean operation  
     * @param groundTruthLayer Expected ground truth layer
     * @param filename Output image filename
     * @param useRawContours If true, shows raw contours; if false, shows simplified polygons
     * @return true if successful
     */
    bool visualizeComprehensiveComparison(const Layer& subjectLayer, const Layer& clipperLayer,
                                        const Layer& resultLayer, const Layer& groundTruthLayer,
                                        const std::string& filename, bool useRawContours = false);

private:
    class EngineImpl;
    std::unique_ptr<EngineImpl> impl;
};

} // namespace GpuLithoLib