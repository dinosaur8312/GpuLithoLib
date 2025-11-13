#include "../include/GpuLithoLib.h"
#include <iostream>

using namespace GpuLithoLib;

int main() {
    std::cout << "=== GPU Contour Tracing Test: INTERSECTION Operation ===\n\n";

    // Initialize GPU device
    int deviceID = 0;
#ifdef __HIP_PLATFORM_AMD__
    hipSetDevice(deviceID);
#else
    cudaSetDevice(deviceID);
#endif
    std::cout << "Using GPU device: " << deviceID << "\n\n";

    // Create lithography engine
    GpuLithoEngine engine(10000, 10000);  // 10000x10000 working area
    engine.enableProfiling(true);

    std::cout << "=== Loading Real Test Data ===\n";

    // Load real test data
    Layer subjectLayer = engine.createLayerFromFile("../testdata/synopsys_simplified_fixed.txt", 0);
    Layer clipperLayer = engine.createLayerFromFile("../testdata/synopsys_simplified_fixed.txt", 1);

    if (!subjectLayer.empty() && !clipperLayer.empty()) {
        std::cout << "Subject layer: " << subjectLayer.getPolygonCount() << " polygons, "
                  << subjectLayer.getVertexCount() << " vertices\n";
        std::cout << "Clipper layer: " << clipperLayer.getPolygonCount() << " polygons, "
                  << clipperLayer.getVertexCount() << " vertices\n\n";

        std::cout << "=== Performing Intersection ===\n";
        // Perform intersection
        Layer result = engine.layerIntersection(subjectLayer, clipperLayer);

        std::cout << "Intersection result: " << result.getPolygonCount() << " polygons, "
                  << result.getVertexCount() << " vertices\n\n";

        std::cout << "=== Generating Visualizations ===\n";
        // Visualize
        engine.visualizeLayer(subjectLayer, "intersection_subject.png", "Subject Layer");
        engine.visualizeLayer(clipperLayer, "intersection_clipper.png", "Clipper Layer");
        engine.visualizeLayer(result, "intersection_result.png", "Intersection Result");

        std::cout << "Generated: intersection_subject.png, intersection_clipper.png, intersection_result.png\n";

        // Load ground truth and compare
        std::cout << "\n=== Ground Truth Comparison ===\n";
        Layer groundTruth = engine.createGroundTruthLayer(
            "../testdata/synopsys_simplified_fixed.txt",
            OperationType::INTERSECTION);

        if (!groundTruth.empty()) {
            engine.visualizeVerificationComparison(result, groundTruth,
                "intersection_verification.png");
            std::cout << "Generated: intersection_verification.png (comparison with ground truth)\n";
        }

        std::cout << "\n=== GPU vs OpenCV Contour Comparison ===\n";
        std::cout << "Generated: contour_comparison_intersection.png\n";
        std::cout << "Color legend:\n";
        std::cout << "  - GREEN pixels: OpenCV findContours only\n";
        std::cout << "  - RED pixels: GPU tracing only\n";
        std::cout << "  - YELLOW pixels: Both methods agree\n";
    } else {
        std::cout << "ERROR: Could not load test data file\n";
        std::cout << "Expected file: ../testdata/synopsys_simplified_fixed.txt\n";
        return 1;
    }

    std::cout << "\n=== Performance Statistics ===\n";
    engine.printPerformanceStats();

    std::cout << "\n=== Test Complete ===\n";

    return 0;
}
