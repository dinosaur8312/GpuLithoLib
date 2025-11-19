#include "GpuKernelProfiler.cuh"

namespace GpuLithoLib {

GpuKernelProfiler::GpuKernelProfiler()
    : rayCastingTime(0.0f), overlayTime(0.0f), intersectionComputeTime(0.0f),
      extractContoursTime(0.0f), traceContoursTime(0.0f) {}

void GpuKernelProfiler::addRayCastingTime(float ms) {
    rayCastingTime += ms;
}

void GpuKernelProfiler::addOverlayTime(float ms) {
    overlayTime += ms;
}

void GpuKernelProfiler::addIntersectionComputeTime(float ms) {
    intersectionComputeTime += ms;
}

void GpuKernelProfiler::addExtractContoursTime(float ms) {
    extractContoursTime += ms;
}

void GpuKernelProfiler::addTraceContoursTime(float ms) {
    traceContoursTime += ms;
}

float GpuKernelProfiler::getTotalTime() const {
    return rayCastingTime + overlayTime + intersectionComputeTime +
           extractContoursTime + traceContoursTime;
}

void GpuKernelProfiler::printSummary() const {
    std::cout << "\n============================================" << std::endl;
    std::cout << "GPU Kernel Timing Summary" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  Ray Casting:            " << rayCastingTime << " ms" << std::endl;
    std::cout << "  Overlay:                " << overlayTime << " ms" << std::endl;
    std::cout << "  Intersection Compute:   " << intersectionComputeTime << " ms" << std::endl;
    std::cout << "  Extract Contours:       " << extractContoursTime << " ms" << std::endl;
    std::cout << "  Trace Contours:         " << traceContoursTime << " ms" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "  Total GPU Kernel Time:  " << getTotalTime() << " ms" << std::endl;
    std::cout << "============================================\n" << std::endl;
}

void GpuKernelProfiler::reset() {
    rayCastingTime = 0.0f;
    overlayTime = 0.0f;
    intersectionComputeTime = 0.0f;
    extractContoursTime = 0.0f;
    traceContoursTime = 0.0f;
}

} // namespace GpuLithoLib
