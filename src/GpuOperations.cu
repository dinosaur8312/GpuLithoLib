#include "GpuOperations.cuh"
#include <cstdio>

namespace GpuLithoLib {

using gpuLitho::OperationType;

#define EPSILON 1e-6

// Ray casting kernel - simplified version from original
__global__ void rayCasting_kernel(
    const uint2* vertices,
    const unsigned int* startIndices,
    const unsigned int* ptCounts,
    const uint4* boxes,
    unsigned int* bitmap,
    const int bitmapWidth,
    const int bitmapHeight,
    const unsigned int polygonNum)
{
    __shared__ uint4 box;
    __shared__ unsigned int boxWidth;
    __shared__ unsigned int boxHeight;
    __shared__ uint2 s_vertices[512];

    int polygonIdx = blockIdx.x;
    if (polygonIdx >= polygonNum) return;
    
    int startIdx = startIndices[polygonIdx];
    int ptCount = ptCounts[polygonIdx];

    if (threadIdx.x == 0) {
        box = boxes[polygonIdx];
        boxWidth = box.z - box.x;
        boxHeight = box.w - box.y;
        
    }
    if (ptCount > 512) {
        printf("Warning: rayCasting_kernel, polygonIdx: %d, ptCount: %d exceeds limit\n", polygonIdx, ptCount);
        return;
    }

    if (threadIdx.x < ptCount) {
        s_vertices[threadIdx.x] = vertices[startIdx + threadIdx.x];
    }
    __syncthreads();

    // Process pixels in bounding box
    int iter_max = iDivUp(boxWidth * boxHeight, blockDim.x);

    for (int iter = 0; iter < iter_max; iter++) {
        int tid = iter * blockDim.x + threadIdx.x;
        int ix = tid % boxWidth + box.x;
        int iy = tid / boxWidth + box.y;

        if (ix >= bitmapWidth || iy >= bitmapHeight || tid >= (boxWidth * boxHeight)) {
            continue;
        }

        // Ray casting algorithm
        bool isInside = false;
        for (int i = 0; i < ptCount; i++) {
            uint2 u_v1 = s_vertices[i];
            uint2 u_v2 = s_vertices[(i + 1) % ptCount];

            int2 v1 = make_int2(u_v1.x, u_v1.y);
            int2 v2 = make_int2(u_v2.x, u_v2.y);

            if ((v1.y <= iy && v2.y > iy) || (v2.y <= iy && v1.y > iy)) {
                float intersectX = v1.x + (float)(iy - v1.y) / (v2.y - v1.y) * (v2.x - v1.x);
                if (intersectX <= (ix + EPSILON)) {
                    isInside = !isInside;
                }
            }
        }

        if (isInside) {
            bitmap[iy * bitmapWidth + ix] = polygonIdx + 1;
        }
    }
}

// Optimized scanline ray casting kernel with edge-based skipping
__global__ void rayCasting_scanline_kernel(
    const uint2* vertices,
    const unsigned int* startIndices,
    const unsigned int* ptCounts,
    const uint4* boxes,
    const unsigned int* edgeBitmap,
    unsigned int* bitmap,
    const int bitmapWidth,
    const int bitmapHeight,
    const unsigned int polygonNum)
{
    // Thread organization: 32 threads per warp (x), 16 rows (y) = 512 threads per block
    const int WARP_SIZE = 32;
    const int ROWS_PER_BLOCK = 16;

    __shared__ uint4 box;
    __shared__ unsigned int boxWidth;
    __shared__ unsigned int boxHeight;
    __shared__ uint2 s_vertices[512];
    __shared__ bool rowState[ROWS_PER_BLOCK];  // Current inside/outside state per row

    int polygonIdx = blockIdx.x;
    if (polygonIdx >= polygonNum) return;

    int startIdx = startIndices[polygonIdx];
    int ptCount = ptCounts[polygonIdx];

    // Load bounding box
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        box = boxes[polygonIdx];
        boxWidth = box.z - box.x;
        boxHeight = box.w - box.y;
    }

    if (ptCount > 512) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("Warning: rayCasting_scanline_kernel, polygonIdx: %d, ptCount: %d exceeds limit\n",
                   polygonIdx, ptCount);
        }
        return;
    }

    // Load vertices to shared memory
    int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    if (tid < ptCount) {
        s_vertices[tid] = vertices[startIdx + tid];
    }
    __syncthreads();

    int laneId = threadIdx.x;       // 0-31 within warp
    int rowInBlock = threadIdx.y;   // 0-15

    // Process bounding box in tiles
    for (int tileY = 0; tileY < boxHeight; tileY += ROWS_PER_BLOCK) {
        int globalY = box.y + tileY + rowInBlock;

        if (globalY >= box.w || globalY >= bitmapHeight) continue;

        // Initialize row state: start outside polygon
        if (laneId == 0) {
            rowState[rowInBlock] = false;
        }
        __syncthreads();

        // Scan along X direction in warp-sized chunks
        for (int tileX = 0; tileX < boxWidth; tileX += WARP_SIZE) {
            int globalX = box.x + tileX + laneId;

            if (globalX >= box.z || globalX >= bitmapWidth) continue;

            int pixelIdx = globalY * bitmapWidth + globalX;

            // Step 1: Check which pixels in warp are edge pixels
            bool isEdge = (edgeBitmap[pixelIdx] == (polygonIdx + 1));
            unsigned int edgeMask = __ballot_sync(0xFFFFFFFF, isEdge);

            // Step 2: Process warp based on edge pattern
            if (edgeMask == 0) {
                // No edges in this warp - simple state propagation
                if (rowState[rowInBlock]) {
                    bitmap[pixelIdx] = polygonIdx + 1;
                }
            }
            else {
                // Has edges - need to process transitions

                // Find first and last edge positions in warp
                int firstEdge = __ffs(edgeMask) - 1;  // Returns 0-31
                int lastEdge = 31 - __clz(edgeMask);

                // Fill pixels before first edge using current state
                if (laneId < firstEdge) {
                    if (rowState[rowInBlock]) {
                        bitmap[pixelIdx] = polygonIdx + 1;
                    }
                }

                // Process edge regions - need to determine state after edges
                // Strategy: find consecutive edge runs, test pixel after last edge in each run
                if (isEdge) {
                    // Check if this is the last edge in a consecutive run
                    bool isLastInRun = (laneId == 31) || !((edgeMask >> (laneId + 1)) & 1);

                    if (isLastInRun && globalX + 1 < bitmapWidth) {
                        // Perform ray casting for pixel immediately after this edge
                        int testX = globalX + 1;
                        int testY = globalY;

                        // Only check edges whose Y range contains testY
                        bool isInside = false;
                        for (int i = 0; i < ptCount; i++) {
                            int2 v1 = make_int2(s_vertices[i].x, s_vertices[i].y);
                            int2 v2 = make_int2(s_vertices[(i + 1) % ptCount].x,
                                               s_vertices[(i + 1) % ptCount].y);

                            // Early skip: check if edge Y range contains testY
                            int ymin = min(v1.y, v2.y);
                            int ymax = max(v1.y, v2.y);
                            if (testY < ymin || testY > ymax) continue;

                            // Standard ray casting check
                            if ((v1.y <= testY && v2.y > testY) ||
                                (v2.y <= testY && v1.y > testY)) {
                                float intersectX = v1.x + (float)(testY - v1.y) /
                                                   (v2.y - v1.y) * (v2.x - v1.x);
                                if (intersectX <= (testX + EPSILON)) {
                                    isInside = !isInside;
                                }
                            }
                        }

                        // Update row state for following pixels
                        // This state will be used by pixels after this edge run
                        rowState[rowInBlock] = isInside;
                    }
                }

                // Synchronize to ensure state is updated
                __syncwarp();

                // Fill pixels between edge runs and after last edge
                // Need to handle multiple edge runs within the warp
                if (laneId > lastEdge) {
                    // Pixels after all edges use the final state
                    if (rowState[rowInBlock]) {
                        bitmap[pixelIdx] = polygonIdx + 1;
                    }
                } else if (laneId > firstEdge && !isEdge) {
                    // Pixel is between edges - determine which state to use
                    // Find the nearest preceding edge that ended a run
                    unsigned int precedingEdges = edgeMask & ((1u << laneId) - 1);

                    if (precedingEdges != 0) {
                        // There are edges before this pixel
                        // Use current row state (updated by last edge run)
                        if (rowState[rowInBlock]) {
                            bitmap[pixelIdx] = polygonIdx + 1;
                        }
                    }
                }
            }

            // Ensure all threads complete before moving to next warp tile
            __syncwarp();
        }

        // Sync between row processing
        __syncthreads();
    }
}





// Edge rendering kernel
__global__ void edgeRender_kernel(
    const uint2* vertices,
    const unsigned int* startIndices,
    const unsigned int* ptCounts,
    unsigned int* bitmap,
    const int bitmapWidth,
    const int bitmapHeight,
    const int mode)
{
    int polygonIdx = blockIdx.x;
    int startIdx = startIndices[polygonIdx];
    int ptCount = ptCounts[polygonIdx];

    for (int i = 0; i < ptCount; i++) {
        uint2 v1 = vertices[startIdx + i];
        uint2 v2 = vertices[startIdx + (i + 1) % ptCount];

        int steps = max(abs((int)v2.x - (int)v1.x), abs((int)v2.y - (int)v1.y));
        if (steps == 0) continue;

        float dx = (float)((int)v2.x - (int)v1.x) / steps;
        float dy = (float)((int)v2.y - (int)v1.y) / steps;

        for (int istep = threadIdx.x; istep < steps; istep += blockDim.x) {
            float x = v1.x + istep * dx;
            float y = v1.y + istep * dy;

            int ix = roundf(x);
            int iy = roundf(y);

            if (ix >= 0 && ix < bitmapWidth && iy >= 0 && iy < bitmapHeight) {
                if (mode == 1) {
                    bitmap[iy * bitmapWidth + ix] = polygonIdx + 1;
                } else {
                    bitmap[iy * bitmapWidth + ix] = 0;
                }
            }
        }
    }
}

// Overlay kernel - combines two bitmaps
__global__ void overlay_kernel(
    const unsigned int* subjectBitmap,
    const unsigned int* clipperBitmap,
    unsigned int* outputBitmap,
    int width,
    int height)
{
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row >= height) return;

    for (int x = col; x < width; x += blockDim.x) {
        int idx = row * width + x;
        unsigned int subjectValue = subjectBitmap[idx];
        unsigned int clipperValue = clipperBitmap[idx];
        
        // Combine: lower 16 bits = subject, upper 16 bits = clipper
        outputBitmap[idx] = subjectValue | (clipperValue << 16);
    }
}

// Simplified offset kernel (basic implementation)
__global__ void offset_kernel(
    const uint2* vertices,
    const unsigned int* startIndices,
    const unsigned int* ptCounts,
    unsigned int* bitmap,
    const int bitmapWidth,
    const int bitmapHeight,
    const int offsetDistance,
    const bool positiveOffset)
{
    // Simplified offset - just dilate/erode the bitmap
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = bitmapWidth * bitmapHeight;
    
    if (idx >= totalPixels) return;
    
    int x = idx % bitmapWidth;
    int y = idx / bitmapWidth;
    
    // Simple morphological operation
    bool hasNeighbor = false;
    for (int dy = -offsetDistance; dy <= offsetDistance; dy++) {
        for (int dx = -offsetDistance; dx <= offsetDistance; dx++) {
            if (dx*dx + dy*dy <= offsetDistance*offsetDistance) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < bitmapWidth && ny >= 0 && ny < bitmapHeight) {
                    int nidx = ny * bitmapWidth + nx;
                    if (bitmap[nidx] > 0) {
                        hasNeighbor = true;
                        break;
                    }
                }
            }
        }
        if (hasNeighbor) break;
    }
    
    // Update pixel based on operation
    if (positiveOffset && hasNeighbor) {
        bitmap[idx] = 1; // Dilate
    } else if (!positiveOffset && !hasNeighbor && bitmap[idx] > 0) {
        bitmap[idx] = 0; // Erode
    }
}

// Note: Device helper functions (calculateDistanceThreshold_device, computeLineIntersection_device)
// are now defined inline in GpuOperations.cuh to support cross-compilation unit calls in HIP

// =============================================================================
// New Scanline Ray Casting Kernels (Range-based approach)
// =============================================================================

// Step 2: Check edge-right neighbor pixels and mark inside/outside status
// For each edge pixel, checks if (x+1, y) is inside or outside the polygon
// Edge bitmap encoding after this kernel:
//   lower 16 bits = polygonIdx+1 (edge marker from edgeRender)
//   upper 16 bits = 0xFFFF if right neighbor is inside, 0x00FF if outside
__global__ void checkEdgeRightNeighbor_kernel(
    const uint2* vertices,
    const unsigned int* startIndices,
    const unsigned int* ptCounts,
    const uint4* boxes,
    unsigned int* edgeBitmap,
    const int bitmapWidth,
    const int bitmapHeight,
    const unsigned int polygonNum)
{
    __shared__ uint2 s_vertices[512];

    int polygonIdx = blockIdx.x;
    if (polygonIdx >= polygonNum) return;

    int startIdx = startIndices[polygonIdx];
    int ptCount = ptCounts[polygonIdx];
    uint4 box = boxes[polygonIdx];

    // Edge value used by edgeRender_kernel is polygonIdx + 1
    unsigned int edgeMarker = polygonIdx + 1;

    if (ptCount > 512) {
        if (threadIdx.x == 0) {
            printf("Warning: checkEdgeRightNeighbor_kernel, polygonIdx: %d, ptCount: %d exceeds limit\n",
                   polygonIdx, ptCount);
        }
        return;
    }

    // Load vertices to shared memory
    if (threadIdx.x < ptCount) {
        s_vertices[threadIdx.x] = vertices[startIdx + threadIdx.x];
    }
    __syncthreads();

    // Each thread processes edges, checking edge pixels
    for (int i = 0; i < ptCount; i++) {
        uint2 v1 = s_vertices[i];
        uint2 v2 = s_vertices[(i + 1) % ptCount];

        int steps = max(abs((int)v2.x - (int)v1.x), abs((int)v2.y - (int)v1.y));
        if (steps == 0) continue;

        float dx = (float)((int)v2.x - (int)v1.x) / steps;
        float dy = (float)((int)v2.y - (int)v1.y) / steps;

        for (int istep = threadIdx.x; istep < steps; istep += blockDim.x) {
            float x = v1.x + istep * dx;
            float y = v1.y + istep * dy;

            int ix = roundf(x);
            int iy = roundf(y);

            if (ix >= 0 && ix < bitmapWidth && iy >= 0 && iy < bitmapHeight) {
                int pixelIdx = iy * bitmapWidth + ix;
                unsigned int edgeValue = edgeBitmap[pixelIdx];

                // Check if this is our edge pixel (edgeRender uses polygonIdx+1)
                if (edgeValue == edgeMarker) {
                    // Check right neighbor (x+1, y)
                    int testX = ix + 1;
                    int testY = iy;

                    // Skip if right neighbor is out of bounds or beyond box
                    if (testX >= bitmapWidth || testX > (int)box.z) {
                        // Mark as outside (right neighbor out of bounds)
                        atomicOr(&edgeBitmap[pixelIdx], 0x00FF0000);  // Outside marker
                        continue;
                    }

                    int rightNeighborIdx = testY * bitmapWidth + testX;
                    unsigned int rightValue = edgeBitmap[rightNeighborIdx];

                    // If right neighbor is also an edge of same polygon, skip
                    if (rightValue == edgeMarker) {
                        continue;
                    }

                    // Perform ray casting for right neighbor pixel
                    // Only check edges whose Y range contains testY
                    bool isInside = false;
                    for (int j = 0; j < ptCount; j++) {
                        int2 ev1 = make_int2(s_vertices[j].x, s_vertices[j].y);
                        int2 ev2 = make_int2(s_vertices[(j + 1) % ptCount].x,
                                            s_vertices[(j + 1) % ptCount].y);

                        // Early skip: check if edge Y range contains testY
                        int ymin = min(ev1.y, ev2.y);
                        int ymax = max(ev1.y, ev2.y);
                        if (testY < ymin || testY > ymax) continue;

                        // Standard ray casting check
                        if ((ev1.y <= testY && ev2.y > testY) ||
                            (ev2.y <= testY && ev1.y > testY)) {
                            float intersectX = ev1.x + (float)(testY - ev1.y) /
                                               (ev2.y - ev1.y) * (ev2.x - ev1.x);
                            if (intersectX <= (testX + EPSILON)) {
                                isInside = !isInside;
                            }
                        }
                    }

                    // Mark edge pixel with inside/outside status in upper 16 bits
                    if (isInside) {
                        atomicOr(&edgeBitmap[pixelIdx], 0xFFFF0000);  // Inside marker
                    } else {
                        atomicOr(&edgeBitmap[pixelIdx], 0x00FF0000);  // Outside marker
                    }
                }
            }
        }
    }
}

// Step 3: Find inside ranges for each scanline
// Each block processes one polygon, each thread processes one scanline
__global__ void findScanlineRanges_kernel(
    const unsigned int* edgeBitmap,
    const uint4* boxes,
    const unsigned int* scanlineOffsets,
    uint2* scanlineRanges,
    unsigned int* scanlineRangeCounts,
    const int bitmapWidth,
    const int bitmapHeight,
    const unsigned int polygonNum,
    const unsigned int maxRangesPerScanline)
{
    int polygonIdx = blockIdx.x;
    if (polygonIdx >= polygonNum) return;

    uint4 box = boxes[polygonIdx];
    unsigned int edgeMarker = polygonIdx + 1;  // Same as edgeRender
    unsigned int boxHeight = box.w - box.y;
    unsigned int scanlineBase = scanlineOffsets[polygonIdx];

    // Each thread handles one scanline within the bounding box
    for (int localY = threadIdx.x; localY < boxHeight; localY += blockDim.x) {
        int globalY = box.y + localY;
        if (globalY >= bitmapHeight) continue;

        unsigned int scanlineIdx = scanlineBase + localY;
        unsigned int rangeCount = 0;
        uint2* rangeOutput = scanlineRanges + scanlineIdx * maxRangesPerScanline;

        bool insideRegion = false;
        unsigned int rangeStart = 0;

        // Scan from left to right within bounding box
        for (int localX = 0; localX <= (int)(box.z - box.x); localX++) {
            int globalX = box.x + localX;
            if (globalX >= bitmapWidth) break;

            int pixelIdx = globalY * bitmapWidth + globalX;
            unsigned int edgeValue = edgeBitmap[pixelIdx];
            unsigned int lowerBits = edgeValue & 0xFFFF;

            if (lowerBits == edgeMarker) {
                // This is an edge pixel of our polygon
                if (insideRegion) {
                    // End current inside region at this edge
                    if (rangeCount < maxRangesPerScanline) {
                        rangeOutput[rangeCount] = make_uint2(rangeStart, globalX);
                        rangeCount++;
                    }
                    insideRegion = false;
                }

                // Check if right neighbor is inside (0xFFFF in upper 16 bits)
                unsigned int upperBits = edgeValue >> 16;
                if (upperBits == 0xFFFF) {
                    // Right neighbor is inside, start new region after this edge
                    insideRegion = true;
                    rangeStart = globalX + 1;
                }
            }
        }

        // Close any open region at end of scanline
        if (insideRegion && rangeCount < maxRangesPerScanline) {
            rangeOutput[rangeCount] = make_uint2(rangeStart, box.z);
            rangeCount++;
        }

        scanlineRangeCounts[scanlineIdx] = rangeCount;
    }
}

// Step 4: Render polygon interior using precomputed scanline ranges
__global__ void renderScanlineRanges_kernel(
    const uint4* boxes,
    const unsigned int* scanlineOffsets,
    const uint2* scanlineRanges,
    const unsigned int* scanlineRangeCounts,
    unsigned int* bitmap,
    const int bitmapWidth,
    const int bitmapHeight,
    const unsigned int polygonNum,
    const unsigned int maxRangesPerScanline)
{
    int polygonIdx = blockIdx.x;
    if (polygonIdx >= polygonNum) return;

    uint4 box = boxes[polygonIdx];
    unsigned int polygonId = polygonIdx + 1;
    unsigned int boxHeight = box.w - box.y;
    unsigned int scanlineBase = scanlineOffsets[polygonIdx];

    // Each thread handles one scanline
    for (int localY = threadIdx.x; localY < boxHeight; localY += blockDim.x) {
        int globalY = box.y + localY;
        if (globalY >= bitmapHeight) continue;

        unsigned int scanlineIdx = scanlineBase + localY;
        unsigned int rangeCount = scanlineRangeCounts[scanlineIdx];
        const uint2* ranges = scanlineRanges + scanlineIdx * maxRangesPerScanline;

        // Fill all ranges for this scanline
        for (unsigned int r = 0; r < rangeCount; r++) {
            uint2 range = ranges[r];
            for (unsigned int x = range.x; x < range.y; x++) {
                if (x < bitmapWidth) {
                    bitmap[globalY * bitmapWidth + x] = polygonId;
                }
            }
        }
    }
}

} // namespace GpuLithoLib
