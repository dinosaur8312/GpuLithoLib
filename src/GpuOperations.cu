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

// Contour extraction kernel
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

    // Check if pixel is contour (has non-ROI neighbors)
    if (s_isContourPixel[threadIdx.y][threadIdx.x] > 0) {
        if ((s_isContourPixel[threadIdx.y - 1][threadIdx.x] == 0) ||
            (s_isContourPixel[threadIdx.y][threadIdx.x - 1] == 0) ||
            (s_isContourPixel[threadIdx.y + 1][threadIdx.x] == 0) ||
            (s_isContourPixel[threadIdx.y][threadIdx.x + 1] == 0)) {
            
            contourBitmap[g_idx] = s_isContourPixel[threadIdx.y][threadIdx.x];
        }
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

// Device helper function to calculate distance threshold based on intersection angle
__device__ float calculateDistanceThreshold_device(float angle_degrees) {
    // Clamp angle to valid range
    angle_degrees = fmaxf(0.0f, fminf(90.0f, angle_degrees));

    // For perpendicular intersections (90 degrees), use minimum threshold
    if (angle_degrees >= 89.0f) {
        return 2.0f;
    }

    // For very acute angles (<= 1 degree), use maximum threshold
    if (angle_degrees <= 1.0f) {
        return 12.0f;
    }

    // Logarithmic interpolation for angles between 1 and 89 degrees
    float log_angle = logf(angle_degrees);
    float log_max = logf(89.0f);
    float normalized = 1.0f - (log_angle / log_max);

    float threshold = 2.0f + 10.0f * normalized;

    // Clamp to [2.0, 12.0] for safety
    return fmaxf(2.0f, fminf(12.0f, threshold));
}

// Device helper function to compute line segment intersection
__device__ bool computeLineIntersection_device(
    float p1x, float p1y, float p2x, float p2y,
    float p3x, float p3y, float p4x, float p4y,
    float& intersectX, float& intersectY, float& angle_degrees) {

    float denom = (p1x - p2x) * (p3y - p4y) - (p1y - p2y) * (p3x - p4x);

    if (fabsf(denom) < EPSILON) {
        return false; // Lines are parallel
    }

    float t = ((p1x - p3x) * (p3y - p4y) - (p1y - p3y) * (p3x - p4x)) / denom;
    float u = -((p1x - p2x) * (p1y - p3y) - (p1y - p2y) * (p1x - p3x)) / denom;

    if (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f) {
        intersectX = p1x + t * (p2x - p1x);
        intersectY = p1y + t * (p2y - p1y);

        // Calculate angle between the two line segments
        float v1x = p2x - p1x;
        float v1y = p2y - p1y;
        float v2x = p4x - p3x;
        float v2y = p4y - p3y;

        float dot = v1x * v2x + v1y * v2y;
        float norm1 = sqrtf(v1x * v1x + v1y * v1y);
        float norm2 = sqrtf(v2x * v2x + v2y * v2y);

        if (norm1 > EPSILON && norm2 > EPSILON) {
            float cos_angle = dot / (norm1 * norm2);
            cos_angle = fmaxf(-1.0f, fminf(1.0f, cos_angle)); // Clamp to avoid NaN

            float angle_rad = acosf(fabsf(cos_angle));
            angle_degrees = angle_rad * 180.0f / M_PI;

            // Ensure we get the acute angle
            if (angle_degrees > 90.0f) {
                angle_degrees = 180.0f - angle_degrees;
            }
        }

        return true;
    }

    return false;
}

// Kernel to compute intersection points for polygon pairs
// Each block handles one intersecting pair
// Assumes: 1) polygons have <= 1024 vertices, 2) max 32 intersection points per pair
__global__ void computeIntersectionPoints_kernel(
    const unsigned int* packedPairs,
    const unsigned int numPairs,
    const uint2* subjectVertices,
    const unsigned int* subjectStartIndices,
    const unsigned int* subjectPtCounts,
    const uint2* clipperVertices,
    const unsigned int* clipperStartIndices,
    const unsigned int* clipperPtCounts,
    IntersectionPointData* outputPoints,
    unsigned int* outputCounts) {

    const unsigned int MAX_INTERSECTIONS = 32;
    const unsigned int pairIdx = blockIdx.x;

    if (pairIdx >= numPairs) return;

    // Unpack the pair
    unsigned int packed = packedPairs[pairIdx];
    unsigned int subject_id = (packed & 0xFFFF) - 1;  // Convert to 0-based
    unsigned int clipper_id = ((packed >> 16) & 0xFFFF) - 1;

    // Load polygon metadata
    unsigned int subject_start = subjectStartIndices[subject_id];
    unsigned int subject_count = subjectPtCounts[subject_id];
    unsigned int clipper_start = clipperStartIndices[clipper_id];
    unsigned int clipper_count = clipperPtCounts[clipper_id];

    if (subject_count == 0 || clipper_count == 0 || subject_count > 1024 || clipper_count > 1024) {
        if (threadIdx.x == 0) {
            outputCounts[pairIdx] = 0;
        }
        return;
    }

    // Shared memory for vertices (assuming max 1024 vertices per polygon)
    __shared__ uint2 s_subject_verts[1024];
    __shared__ uint2 s_clipper_verts[1024];
    __shared__ unsigned int s_intersection_count;

    // Initialize shared intersection count
    if (threadIdx.x == 0) {
        s_intersection_count = 0;
    }
    __syncthreads();

    // Load subject vertices into shared memory
    for (unsigned int i = threadIdx.x; i < subject_count; i += blockDim.x) {
        s_subject_verts[i] = subjectVertices[subject_start + i];
    }

    // Load clipper vertices into shared memory
    for (unsigned int i = threadIdx.x; i < clipper_count; i += blockDim.x) {
        s_clipper_verts[i] = clipperVertices[clipper_start + i];
    }

    __syncthreads();

    // Each thread computes intersections for a subset of edge pairs
    // Total edge pairs = subject_count * clipper_count
    unsigned int totalEdgePairs = subject_count * clipper_count;

    for (unsigned int edgePairIdx = threadIdx.x; edgePairIdx < totalEdgePairs; edgePairIdx += blockDim.x) {
        unsigned int si = edgePairIdx / clipper_count;
        unsigned int ci = edgePairIdx % clipper_count;

        unsigned int next_si = (si + 1) % subject_count;
        unsigned int next_ci = (ci + 1) % clipper_count;

        // Get edge vertices
        float s1x = static_cast<float>(s_subject_verts[si].x);
        float s1y = static_cast<float>(s_subject_verts[si].y);
        float s2x = static_cast<float>(s_subject_verts[next_si].x);
        float s2y = static_cast<float>(s_subject_verts[next_si].y);

        float c1x = static_cast<float>(s_clipper_verts[ci].x);
        float c1y = static_cast<float>(s_clipper_verts[ci].y);
        float c2x = static_cast<float>(s_clipper_verts[next_ci].x);
        float c2y = static_cast<float>(s_clipper_verts[next_ci].y);

        // Compute intersection
        float intersectX, intersectY, angle_degrees = 90.0f;

        if (computeLineIntersection_device(s1x, s1y, s2x, s2y, c1x, c1y, c2x, c2y,
                                           intersectX, intersectY, angle_degrees)) {
            // Calculate distance threshold based on angle
            float threshold = calculateDistanceThreshold_device(angle_degrees);

            // Atomically get slot for this intersection
            unsigned int slot = atomicAdd(&s_intersection_count, 1);

            if (slot < MAX_INTERSECTIONS) {
                // Store in global output
                unsigned int outputIdx = pairIdx * MAX_INTERSECTIONS + slot;
                outputPoints[outputIdx].x = static_cast<unsigned int>(roundf(intersectX));
                outputPoints[outputIdx].y = static_cast<unsigned int>(roundf(intersectY));
                outputPoints[outputIdx].distanceThreshold = threshold;
            }
        }
    }

    __syncthreads();

    // Write final count
    if (threadIdx.x == 0) {
        outputCounts[pairIdx] = min(s_intersection_count, MAX_INTERSECTIONS);
    }
}

} // namespace GpuLithoLib