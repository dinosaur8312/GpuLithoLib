#pragma once

#include "../include/gpuLitho.h"
#include <iostream>

namespace GpuLithoLib {

// GpuKernelProfiler: Tracks GPU kernel execution times
class GpuKernelProfiler {
public:
    GpuKernelProfiler();
    ~GpuKernelProfiler() = default;

    // Add timing for each kernel type
    void addRayCastingTime(float ms);
    void addOverlayTime(float ms);
    void addIntersectionComputeTime(float ms);
    void addExtractContoursTime(float ms);
    void addTraceContoursTime(float ms);

    // Print timing summary
    void printSummary() const;

    // Reset all timers
    void reset();

    // Get individual times
    float getRayCastingTime() const { return rayCastingTime; }
    float getOverlayTime() const { return overlayTime; }
    float getIntersectionComputeTime() const { return intersectionComputeTime; }
    float getExtractContoursTime() const { return extractContoursTime; }
    float getTraceContoursTime() const { return traceContoursTime; }
    float getTotalTime() const;

private:
    float rayCastingTime;
    float overlayTime;
    float intersectionComputeTime;
    float extractContoursTime;
    float traceContoursTime;
};

} // namespace GpuLithoLib
