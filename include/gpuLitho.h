#ifndef GPULITHO_H
#define GPULITHO_H

// Platform-agnostic header that works with both CUDA and HIP

#ifdef __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
    #include <hip/hip_vector_types.h>

    // HIP type definitions
    #define gpuError_t hipError_t
    #define gpuSuccess hipSuccess
    #define gpuGetErrorString hipGetErrorString
    #define gpuGetLastError hipGetLastError
    #define gpuDeviceSynchronize hipDeviceSynchronize
    #define gpuMemcpy hipMemcpy
    #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
    #define gpuMalloc hipMalloc
    #define gpuFree hipFree
    #define gpuMemset hipMemset
    #define gpuMemGetInfo hipMemGetInfo
    #define gpuSetDevice hipSetDevice

    // Event definitions
    #define gpuEvent_t hipEvent_t
    #define gpuEventCreate hipEventCreate
    #define gpuEventRecord hipEventRecord
    #define gpuEventSynchronize hipEventSynchronize
    #define gpuEventElapsedTime hipEventElapsedTime
    #define gpuEventDestroy hipEventDestroy

    // Warp functions
    #define __syncwarp(mask) __syncthreads()

    // Vector type helpers (HIP uses the same names as CUDA)
    // Just ensure they're available
    using ::uint2;
    using ::uint4;
    using ::int2;
    using ::make_uint2;
    using ::make_uint4;
    using ::make_int2;

#else
    #include <cuda_runtime.h>
    #include <vector_types.h>

    // CUDA type definitions
    #define gpuError_t cudaError_t
    #define gpuSuccess cudaSuccess
    #define gpuGetErrorString cudaGetErrorString
    #define gpuGetLastError cudaGetLastError
    #define gpuDeviceSynchronize cudaDeviceSynchronize
    #define gpuMemcpy cudaMemcpy
    #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
    #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
    #define gpuMalloc cudaMalloc
    #define gpuFree cudaFree
    #define gpuMemset cudaMemset
    #define gpuMemGetInfo cudaMemGetInfo
    #define gpuSetDevice cudaSetDevice

    // Event definitions
    #define gpuEvent_t cudaEvent_t
    #define gpuEventCreate cudaEventCreate
    #define gpuEventRecord cudaEventRecord
    #define gpuEventSynchronize cudaEventSynchronize
    #define gpuEventElapsedTime cudaEventElapsedTime
    #define gpuEventDestroy cudaEventDestroy

#endif

#include <iostream>

#define EPSILON 1e-6

// Helper function for error checking
inline void checkGpuError(gpuError_t code, const char *file, int line) {
    if (code != gpuSuccess) {
        fprintf(stderr, "GPU Error: %s %s %d\n", gpuGetErrorString(code), file, line);
        exit(1);
    }
}

#define CHECK_GPU_ERROR(val) checkGpuError((val), __FILE__, __LINE__)

// Helper math functions that work on both platforms
__host__ __device__ inline int iDivUp(int a, int b) {
    return (a + b - 1) / b;
}

/**
 * @brief Enum for bitmap combination modes in dual bitmap processing
 */
enum class BitmapCombineMode {
    INTERSECTION,  // Both bitmaps must have non-zero values
    UNION,         // Either bitmap can have non-zero values
    DIFFERENCE,    // Subject has non-zero, clipper has zero
    XOR            // Exactly one bitmap has non-zero values
};

#endif // GPULITHO_H
