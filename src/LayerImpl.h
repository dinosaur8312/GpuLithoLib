#pragma once

#include "../include/gpuLitho.h"
#include "../include/operation_types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace GpuLithoLib {

/**
 * @brief Internal implementation of Layer class
 * 
 * This class encapsulates the actual polygon data and GPU memory management.
 * It's based on the existing PolygonLayer but simplified for library use.
 */
class LayerImpl {
public:
    // Constructor
    LayerImpl(unsigned int maxVertices = 1000000, unsigned int maxPolygons = 100000);
    
    // Copy constructor - performs deep copy
    LayerImpl(const LayerImpl& other);
    
    // Destructor
    ~LayerImpl();
    
    // Host data
    uint2* h_vertices;
    unsigned int* h_startIndices;
    unsigned int* h_ptCounts;
    uint4* h_boxes;  // Host bounding boxes

    // Device data
    uint2* d_vertices;
    unsigned int* d_startIndices;
    unsigned int* d_ptCounts;
    uint4* d_boxes;  // Device bounding boxes
    
    // Bitmap for GPU operations
    unsigned int* d_bitmap;
    unsigned int* h_bitmap;
    bool bitmapInitialized;
    
    // Metadata
    unsigned int polygonCount;
    unsigned int vertexCount;
    unsigned int maxVertices;
    unsigned int maxPolygons;
    unsigned int bitmapWidth;
    unsigned int bitmapHeight;
    
    // === Memory Management ===
    void allocateHost();
    void allocateDevice();
    void allocateBitmap(unsigned int width, unsigned int height);
    void freeHost();
    void freeDevice();
    void freeBitmap();
    
    // === Data Transfer ===
    void copyToDevice();
    void copyToHost();
    void copyBitmapToHost();
    void copyBitmapToDevice();
    
    // === Polygon Operations ===
    void addPolygon(const uint2* vertices, unsigned int count);
    void clear();
    bool empty() const { return polygonCount == 0; }
    
    // === Bounding Box ===
    void calculateBoundingBoxes();
    std::vector<unsigned int> getBoundingBox() const;
    
    // === Coordinate Transformation ===
    void shift(int shiftX, int shiftY);
    
    // === File I/O ===
    bool loadFromFile(const std::string& filename, unsigned int layerIndex);
    bool saveToFile(const std::string& filename, unsigned int layerIndex) const;
    
    // === Geometry Generation ===
    void generateRectangle(unsigned int x, unsigned int y, unsigned int width, unsigned int height);
    void generateCircle(unsigned int centerX, unsigned int centerY, unsigned int radius, unsigned int numSides);
    void generateRegularPolygon(unsigned int centerX, unsigned int centerY, unsigned int radius, unsigned int numSides);
    void generateLShape(unsigned int centerX, unsigned int centerY, unsigned int width, unsigned int height, unsigned int thickness);
    
    // === Bitmap Operations ===
    void clearBitmap();
    void visualizeBitmap(const std::string& filename) const;
    void ensureBitmapAllocated(unsigned int width, unsigned int height);
    
private:
    // Helper methods
    void ensureDeviceAllocated();
};

} // namespace GpuLithoLib