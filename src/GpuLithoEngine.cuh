#pragma once

#include "../include/GpuLithoLib.h"
#include "LayerImpl.h"
#include "IntersectionCompute.cuh"
#include "ContourProcessing.cuh"
#include <memory>

namespace GpuLithoLib {

/**
 * @brief Internal implementation of GpuLithoEngine
 * Handles coordinate transformations, layer preparation, and GPU operations
 */
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

    EngineImpl(unsigned int maxW, unsigned int maxH);

    // Layer preparation methods
    void prepareSingleLayer(const Layer& layer);
    void prepareDualLayers(const Layer& layer1, const Layer& layer2);
    void reset();

    // Coordinate transformation
    void restoreOriginalCoordinates(LayerImpl* layer);

    // GPU operation wrappers
    void performRayCasting(LayerImpl* layer, int edgeMode = 1);
    void performOverlay(LayerImpl* subject, LayerImpl* clipper, LayerImpl* output);

    // Contour extraction (simplified version)
    Layer extractContours(LayerImpl* input, OperationType opType);

    // Boolean operations with full pipeline
    Layer performBooleanOperation(OperationType opType);

    // Single-layer operations
    Layer performSingleLayerOperation(OperationType opType, int offsetDistance = 0);
};

} // namespace GpuLithoLib
