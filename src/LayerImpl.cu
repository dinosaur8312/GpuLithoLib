#include "LayerImpl.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace GpuLithoLib {

// GPU kernel for calculating bounding boxes (copied from original)
__global__ void calculatePolygonBoundingBoxes(
    const uint2* vertices,
    const unsigned int* startIndices,
    const unsigned int* ptCounts,
    uint4* boxes,
    const unsigned int polygonNum)
{
    int polygonIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (polygonIdx < polygonNum) {
        unsigned int minX = UINT_MAX;
        unsigned int minY = UINT_MAX;
        unsigned int maxX = 0;
        unsigned int maxY = 0;

        unsigned int startIdx = startIndices[polygonIdx];
        for (int i = 0; i < ptCounts[polygonIdx]; i++) {
            uint2 vertex = vertices[startIdx + i];
            minX = min(minX, vertex.x);
            minY = min(minY, vertex.y);
            maxX = max(maxX, vertex.x);
            maxY = max(maxY, vertex.y);
        }
        boxes[polygonIdx] = {minX, minY, maxX, maxY};

    }
}

LayerImpl::LayerImpl(unsigned int maxVerts, unsigned int maxPolys)
    : h_vertices(nullptr), h_startIndices(nullptr), h_ptCounts(nullptr),
      d_vertices(nullptr), d_startIndices(nullptr), d_ptCounts(nullptr), d_boxes(nullptr),
      d_bitmap(nullptr), h_bitmap(nullptr), bitmapInitialized(false),
      polygonCount(0), vertexCount(0), maxVertices(maxVerts), maxPolygons(maxPolys),
      bitmapWidth(0), bitmapHeight(0)
{
    allocateHost();
}

LayerImpl::LayerImpl(const LayerImpl& other)
    : h_vertices(nullptr), h_startIndices(nullptr), h_ptCounts(nullptr),
      d_vertices(nullptr), d_startIndices(nullptr), d_ptCounts(nullptr), d_boxes(nullptr),
      d_bitmap(nullptr), h_bitmap(nullptr), bitmapInitialized(false),
      polygonCount(other.polygonCount), vertexCount(other.vertexCount),
      maxVertices(other.maxVertices), maxPolygons(other.maxPolygons),
      bitmapWidth(other.bitmapWidth), bitmapHeight(other.bitmapHeight)
{
    allocateHost();
    
    // Deep copy host data
    if (other.h_vertices && vertexCount > 0) {
        memcpy(h_vertices, other.h_vertices, vertexCount * sizeof(uint2));
    }
    if (other.h_startIndices && polygonCount > 0) {
        memcpy(h_startIndices, other.h_startIndices, polygonCount * sizeof(unsigned int));
    }
    if (other.h_ptCounts && polygonCount > 0) {
        memcpy(h_ptCounts, other.h_ptCounts, polygonCount * sizeof(unsigned int));
    }
    
    // Deep copy device data if it exists
    if (other.d_vertices) {
        allocateDevice();
        if (vertexCount > 0) {
            gpuMemcpy(d_vertices, other.d_vertices, vertexCount * sizeof(uint2), gpuMemcpyDeviceToDevice);
        }
        if (polygonCount > 0) {
            gpuMemcpy(d_startIndices, other.d_startIndices, polygonCount * sizeof(unsigned int), gpuMemcpyDeviceToDevice);
            gpuMemcpy(d_ptCounts, other.d_ptCounts, polygonCount * sizeof(unsigned int), gpuMemcpyDeviceToDevice);
        }
        if (other.d_boxes && polygonCount > 0) {
            gpuMemcpy(d_boxes, other.d_boxes, polygonCount * sizeof(uint4), gpuMemcpyDeviceToDevice);
        }
    }
    
    // Deep copy bitmap if it exists
    if (other.bitmapInitialized && bitmapWidth > 0 && bitmapHeight > 0) {
        allocateBitmap(bitmapWidth, bitmapHeight);
        size_t bitmapSize = bitmapWidth * bitmapHeight * sizeof(unsigned int);
        if (other.d_bitmap) {
            gpuMemcpy(d_bitmap, other.d_bitmap, bitmapSize, gpuMemcpyDeviceToDevice);
        }
        if (other.h_bitmap) {
            memcpy(h_bitmap, other.h_bitmap, bitmapSize);
        }
    }
}

LayerImpl::~LayerImpl() {
    freeHost();
    freeDevice();
    freeBitmap();
}

void LayerImpl::allocateHost() {
    if (!h_vertices) {
        h_vertices = new uint2[maxVertices];
        h_startIndices = new unsigned int[maxPolygons];
        h_ptCounts = new unsigned int[maxPolygons];
    }
}

void LayerImpl::allocateDevice() {
    if (!d_vertices) {
        CHECK_GPU_ERROR(gpuMalloc(&d_vertices, maxVertices * sizeof(uint2)));
        CHECK_GPU_ERROR(gpuMalloc(&d_startIndices, maxPolygons * sizeof(unsigned int)));
        CHECK_GPU_ERROR(gpuMalloc(&d_ptCounts, maxPolygons * sizeof(unsigned int)));
        CHECK_GPU_ERROR(gpuMalloc(&d_boxes, maxPolygons * sizeof(uint4)));
    }
}

void LayerImpl::allocateBitmap(unsigned int width, unsigned int height) {
    if (bitmapInitialized && bitmapWidth == width && bitmapHeight == height) {
        return; // Already allocated with correct size
    }
    
    freeBitmap(); // Free existing bitmap if any
    
    bitmapWidth = width;
    bitmapHeight = height;
    size_t bitmapSize = width * height * sizeof(unsigned int);
    
    CHECK_GPU_ERROR(gpuMalloc(&d_bitmap, bitmapSize));
    h_bitmap = new unsigned int[width * height];
    
    // Initialize bitmap to zero
    CHECK_GPU_ERROR(gpuMemset(d_bitmap, 0, bitmapSize));
    memset(h_bitmap, 0, bitmapSize);
    
    bitmapInitialized = true;
}

void LayerImpl::freeHost() {
    delete[] h_vertices;
    delete[] h_startIndices;
    delete[] h_ptCounts;
    h_vertices = nullptr;
    h_startIndices = nullptr;
    h_ptCounts = nullptr;
}

void LayerImpl::freeDevice() {
    if (d_vertices) {
        gpuFree(d_vertices);
        gpuFree(d_startIndices);
        gpuFree(d_ptCounts);
        gpuFree(d_boxes);
        d_vertices = nullptr;
        d_startIndices = nullptr;
        d_ptCounts = nullptr;
        d_boxes = nullptr;
    }
}

void LayerImpl::freeBitmap() {
    if (d_bitmap) {
        gpuFree(d_bitmap);
        d_bitmap = nullptr;
    }
    delete[] h_bitmap;
    h_bitmap = nullptr;
    bitmapInitialized = false;
    bitmapWidth = bitmapHeight = 0;
}

void LayerImpl::copyToDevice() {
    ensureDeviceAllocated();
    if (vertexCount > 0) {
        CHECK_GPU_ERROR(gpuMemcpy(d_vertices, h_vertices, vertexCount * sizeof(uint2), gpuMemcpyHostToDevice));
    }
    if (polygonCount > 0) {
        CHECK_GPU_ERROR(gpuMemcpy(d_startIndices, h_startIndices, polygonCount * sizeof(unsigned int), gpuMemcpyHostToDevice));
        CHECK_GPU_ERROR(gpuMemcpy(d_ptCounts, h_ptCounts, polygonCount * sizeof(unsigned int), gpuMemcpyHostToDevice));
    }
}

void LayerImpl::copyToHost() {
    if (d_vertices && vertexCount > 0) {
        CHECK_GPU_ERROR(gpuMemcpy(h_vertices, d_vertices, vertexCount * sizeof(uint2), gpuMemcpyDeviceToHost));
    }
    if (d_startIndices && polygonCount > 0) {
        CHECK_GPU_ERROR(gpuMemcpy(h_startIndices, d_startIndices, polygonCount * sizeof(unsigned int), gpuMemcpyDeviceToHost));
        CHECK_GPU_ERROR(gpuMemcpy(h_ptCounts, d_ptCounts, polygonCount * sizeof(unsigned int), gpuMemcpyDeviceToHost));
    }
}

void LayerImpl::copyBitmapToHost() {
    if (bitmapInitialized && d_bitmap && h_bitmap) {
        size_t bitmapSize = bitmapWidth * bitmapHeight * sizeof(unsigned int);
        CHECK_GPU_ERROR(gpuMemcpy(h_bitmap, d_bitmap, bitmapSize, gpuMemcpyDeviceToHost));
    }
}

void LayerImpl::copyBitmapToDevice() {
    if (bitmapInitialized && d_bitmap && h_bitmap) {
        size_t bitmapSize = bitmapWidth * bitmapHeight * sizeof(unsigned int);
        CHECK_GPU_ERROR(gpuMemcpy(d_bitmap, h_bitmap, bitmapSize, gpuMemcpyHostToDevice));
    }
}

void LayerImpl::addPolygon(const uint2* vertices, unsigned int count) {
    if (polygonCount >= maxPolygons || vertexCount + count > maxVertices) {
        std::cerr << "Error: Cannot add polygon - would exceed limits" << std::endl;
        return;
    }
    
    h_startIndices[polygonCount] = vertexCount;
    h_ptCounts[polygonCount] = count;
    
    for (unsigned int i = 0; i < count; i++) {
        h_vertices[vertexCount + i] = vertices[i];
    }
    
    vertexCount += count;
    polygonCount++;
}

void LayerImpl::clear() {
    polygonCount = 0;
    vertexCount = 0;
}

void LayerImpl::calculateBoundingBoxes() {
    if (polygonCount == 0) return;
    
    ensureDeviceAllocated();
    copyToDevice();
    
    dim3 blockDim(256);
    dim3 gridDim((polygonCount + blockDim.x - 1) / blockDim.x);
    
    calculatePolygonBoundingBoxes<<<gridDim, blockDim>>>(
        d_vertices, d_startIndices, d_ptCounts, d_boxes, polygonCount);
    
    CHECK_GPU_ERROR(gpuGetLastError());
    CHECK_GPU_ERROR(gpuDeviceSynchronize());
}

std::vector<unsigned int> LayerImpl::getBoundingBox() const {
    if (vertexCount == 0) {
        return {0, 0, 0, 0};
    }
    
    unsigned int minX = UINT_MAX, minY = UINT_MAX;
    unsigned int maxX = 0, maxY = 0;
    
    for (unsigned int i = 0; i < vertexCount; ++i) {
        minX = std::min(minX, h_vertices[i].x);
        minY = std::min(minY, h_vertices[i].y);
        maxX = std::max(maxX, h_vertices[i].x);
        maxY = std::max(maxY, h_vertices[i].y);
    }
    
    return {minX, minY, maxX, maxY};
}

void LayerImpl::shift(int shiftX, int shiftY) {
    if (empty()) return;
    
    for (unsigned int i = 0; i < vertexCount; ++i) {
        // Handle potential underflow by checking bounds
        int newX = static_cast<int>(h_vertices[i].x) + shiftX;
        int newY = static_cast<int>(h_vertices[i].y) + shiftY;
        
        h_vertices[i].x = static_cast<unsigned int>(std::max(0, newX));
        h_vertices[i].y = static_cast<unsigned int>(std::max(0, newY));
    }
    
    // If data is on device, update it
    if (d_vertices) {
        copyToDevice();
    }
}

void LayerImpl::clearBitmap() {
    if (bitmapInitialized && d_bitmap) {
        size_t bitmapSize = bitmapWidth * bitmapHeight * sizeof(unsigned int);
        CHECK_GPU_ERROR(gpuMemset(d_bitmap, 0, bitmapSize));
    }
}

void LayerImpl::ensureDeviceAllocated() {
    if (!d_vertices) {
        allocateDevice();
    }
}

void LayerImpl::ensureBitmapAllocated(unsigned int width, unsigned int height) {
    if (!bitmapInitialized || bitmapWidth != width || bitmapHeight != height) {
        allocateBitmap(width, height);
    }
}

// Geometry generation methods
void LayerImpl::generateRectangle(unsigned int x, unsigned int y, unsigned int width, unsigned int height) {
    uint2 vertices[4] = {
        {x, y},
        {x + width, y},
        {x + width, y + height},
        {x, y + height}
    };
    addPolygon(vertices, 4);
}

void LayerImpl::generateCircle(unsigned int centerX, unsigned int centerY, unsigned int radius, unsigned int numSides) {
    generateRegularPolygon(centerX, centerY, radius, numSides);
}

void LayerImpl::generateRegularPolygon(unsigned int centerX, unsigned int centerY, unsigned int radius, unsigned int numSides) {
    if (numSides < 3) numSides = 3;
    
    std::vector<uint2> vertices(numSides);
    for (unsigned int i = 0; i < numSides; ++i) {
        double angle = 2.0 * M_PI * i / numSides;
        vertices[i].x = centerX + static_cast<unsigned int>(radius * cos(angle));
        vertices[i].y = centerY + static_cast<unsigned int>(radius * sin(angle));
    }
    addPolygon(vertices.data(), numSides);
}

void LayerImpl::generateLShape(unsigned int centerX, unsigned int centerY, unsigned int width, unsigned int height, unsigned int thickness) {
    // Create L-shape as two rectangles
    // Horizontal part
    generateRectangle(centerX - width/2, centerY - thickness/2, width, thickness);
    // Vertical part
    generateRectangle(centerX - thickness/2, centerY - thickness/2, thickness, height);
}

bool LayerImpl::loadFromFile(const std::string& filename, unsigned int layerIndex) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    clear();
    
    std::string line;
    unsigned int currentLayer = 0;
    bool foundTargetLayer = false;
    
    // Skip header lines
    for (int i = 0; i < 8 && std::getline(file, line); ++i);
    
    // Parse polygons
    while (std::getline(file, line)) {
        if (line.find("LAYER") != std::string::npos) {
            std::stringstream ss(line);
            std::string token;
            ss >> token >> currentLayer;
        }
        else if (line.find("XY") != std::string::npos && currentLayer == layerIndex) {
            foundTargetLayer = true;
            std::vector<uint2> vertices;
            while (std::getline(file, line) && line.find("ENDEL") == std::string::npos) {
                // Handle both "x y" and "x: y" formats
                size_t colonPos = line.find(':');
                if (colonPos != std::string::npos) {
                    // Format: "x: y"
                    std::string xStr = line.substr(0, colonPos);
                    std::string yStr = line.substr(colonPos + 1);
                    try {
                        int x = std::stoi(xStr);
                        int y = std::stoi(yStr);
                        vertices.push_back({static_cast<unsigned int>(x), static_cast<unsigned int>(y)});
                    } catch (...) {
                        // Skip invalid lines
                    }
                } else {
                    // Format: "x y"
                    std::stringstream ss(line);
                    int x, y;
                    if (ss >> x >> y) {
                        vertices.push_back({static_cast<unsigned int>(x), static_cast<unsigned int>(y)});
                    }
                }
            }
            if (vertices.size() >= 3) {
                addPolygon(vertices.data(), vertices.size());
            }
        }
    }
    
    return foundTargetLayer && polygonCount > 0;
}

bool LayerImpl::saveToFile(const std::string& filename, unsigned int layerIndex) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write simple header
    file << "GDSII Library\n";
    file << "Generated by GpuLithoLib\n";
    file << "LAYER " << layerIndex << "\n";
    
    // Write polygons
    for (unsigned int i = 0; i < polygonCount; ++i) {
        file << "BOUNDARY\n";
        file << "LAYER " << layerIndex << "\n";
        file << "DATATYPE 0\n";
        file << "XY\n";
        
        unsigned int start = h_startIndices[i];
        unsigned int count = h_ptCounts[i];
        
        for (unsigned int j = 0; j < count; ++j) {
            file << h_vertices[start + j].x << " " << h_vertices[start + j].y << "\n";
        }
        // Close polygon by repeating first vertex
        if (count > 0) {
            file << h_vertices[start].x << " " << h_vertices[start].y << "\n";
        }
        
        file << "ENDEL\n";
    }
    
    file << "ENDSTR\n";
    file << "ENDLIB\n";
    
    return true;
}

void LayerImpl::visualizeBitmap(const std::string& filename) const {
    if (!bitmapInitialized || !h_bitmap) {
        std::cerr << "Warning: No bitmap data to visualize" << std::endl;
        return;
    }
    
    cv::Mat image(bitmapHeight, bitmapWidth, CV_8UC3, cv::Scalar(0, 0, 0)); // Black background
    
    for (unsigned int y = 0; y < bitmapHeight; ++y) {
        for (unsigned int x = 0; x < bitmapWidth; ++x) {
            unsigned int idx = y * bitmapWidth + x;
            if (h_bitmap[idx] > 0) {
                // White pixel for non-zero values
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
            }
        }
    }
    
    cv::imwrite(filename, image);
}

} // namespace GpuLithoLib