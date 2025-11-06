#include "../include/GpuLithoLib.h"
#include <iostream>

using namespace GpuLithoLib;

int main() {
    std::cout << "=== GPU Lithography Library Example ===\n\n";
    
    // Initialize GPU device
    int deviceID = 0;
#ifdef __HIP_PLATFORM_AMD__
    hipSetDevice(deviceID);
#else
    cudaSetDevice(deviceID);
#endif
    std::cout << "Using GPU device: " << deviceID << "\n";
    
    // Create lithography engine
    GpuLithoEngine engine(5000, 5000);  // 5000x5000 working area
    engine.enableProfiling(true);
    
    std::cout << "\n=== Creating Layers ===\n";
    
    // Example 1: Create layers from geometry
    GeometryConfig rectConfig;
    rectConfig.shape = GeometryConfig::RECTANGLE;
    rectConfig.centerX = 1000;
    rectConfig.centerY = 1000; 
    rectConfig.width = 800;
    rectConfig.height = 600;
    rectConfig.gridWidth = 2;   // 2x2 grid
    rectConfig.gridHeight = 2;
    rectConfig.spacingX = 1200;
    rectConfig.spacingY = 1000;
    
    Layer rectangleLayer = engine.createLayerFromGeometry(rectConfig);
    std::cout << "Created rectangle layer: " << rectangleLayer.getPolygonCount() << " polygons, "
              << rectangleLayer.getVertexCount() << " vertices\n";
    
    // Create circle layer
    GeometryConfig circleConfig;
    circleConfig.shape = GeometryConfig::CIRCLE;
    circleConfig.centerX = 1200;
    circleConfig.centerY = 1200;
    circleConfig.radius = 400;
    circleConfig.numSides = 16;  // 16-sided approximation
    circleConfig.gridWidth = 2;
    circleConfig.gridHeight = 2; 
    circleConfig.spacingX = 1200;
    circleConfig.spacingY = 1000;
    
    Layer circleLayer = engine.createLayerFromGeometry(circleConfig);
    std::cout << "Created circle layer: " << circleLayer.getPolygonCount() << " polygons, "
              << circleLayer.getVertexCount() << " vertices\n";
    
    // Example 2: Create layers from test file
    std::cout << "\n=== Loading Test Data ===\n";
    
    // Load subject and clipper layers from test file
    Layer subjectLayer = engine.createLayerFromFile("../testdata/synopsys_simplified_fixed.txt", 0);
    Layer clipperLayer = engine.createLayerFromFile("../testdata/synopsys_simplified_fixed.txt", 1);
    
    if (!subjectLayer.empty() && !clipperLayer.empty()) {
        std::cout << "Subject layer: " << subjectLayer.getPolygonCount() << " polygons, "
                  << subjectLayer.getVertexCount() << " vertices\n";
        std::cout << "Clipper layer: " << clipperLayer.getPolygonCount() << " polygons, "
                  << clipperLayer.getVertexCount() << " vertices\n";
    } else {
        std::cout << "Warning: Could not load test data file\n";
        std::cout << "Using synthetic layers instead\n";
        subjectLayer = rectangleLayer;
        clipperLayer = circleLayer;
    }
    
    std::cout << "\n=== Boolean Operations ===\n";
    
    // Test boolean operations on loaded data
    std::cout << "Testing operations on " 
              << (subjectLayer.getPolygonCount() == rectangleLayer.getPolygonCount() ? "synthetic" : "file") 
              << " layers\n";
    
    // Intersection
    Layer intersectionResult = engine.layerIntersection(subjectLayer, clipperLayer);
    std::cout << "Intersection result: " << intersectionResult.getPolygonCount() << " polygons, "
              << intersectionResult.getVertexCount() << " vertices\n";
    
    // Union
    Layer unionResult = engine.layerUnion(subjectLayer, clipperLayer);
    std::cout << "Union result: " << unionResult.getPolygonCount() << " polygons, "
              << unionResult.getVertexCount() << " vertices\n";
    
    // Difference
    Layer differenceResult = engine.layerDifference(subjectLayer, clipperLayer);
    std::cout << "Difference result: " << differenceResult.getPolygonCount() << " polygons, "
              << differenceResult.getVertexCount() << " vertices\n";
    
    // XOR
    Layer xorResult = engine.layerXor(subjectLayer, clipperLayer);
    std::cout << "XOR result: " << xorResult.getPolygonCount() << " polygons, "
              << xorResult.getVertexCount() << " vertices\n";
    
    std::cout << "\n=== Geometric Operations ===\n";
    
    // Offset operations
    Layer positiveOffset = engine.layerOffset(rectangleLayer, 50);  // Grow by 50 pixels
    std::cout << "Positive offset (+50) result: " << positiveOffset.getPolygonCount() << " polygons\n";
    
    Layer negativeOffset = engine.layerOffset(rectangleLayer, -30); // Shrink by 30 pixels  
    std::cout << "Negative offset (-30) result: " << negativeOffset.getPolygonCount() << " polygons\n";
    
    std::cout << "\n=== Visualization and Output ===\n";
    
    // Create visualizations for input layers
    if (subjectLayer.getPolygonCount() != rectangleLayer.getPolygonCount()) {
        // File layers loaded successfully
        engine.visualizeLayer(subjectLayer, "subject_layer.png", false);
        engine.visualizeLayer(clipperLayer, "clipper_layer.png", false);
        std::cout << "Visualized test file layers\n";
    } else {
        // Synthetic layers
        engine.visualizeLayer(rectangleLayer, "rectangle_layer.png", true);
        engine.visualizeLayer(circleLayer, "circle_layer.png", true);
        std::cout << "Visualized synthetic layers\n";
    }
    
    // Create visualizations for results
    engine.visualizeLayer(intersectionResult, "intersection_result.png");
    engine.visualizeLayer(unionResult, "union_result.png");
    engine.visualizeLayer(differenceResult, "difference_result.png");
    engine.visualizeLayer(xorResult, "xor_result.png");
    engine.visualizeLayer(positiveOffset, "positive_offset.png");
    engine.visualizeLayer(negativeOffset, "negative_offset.png");
    std::cout << "Visualization images saved\n";
    
    // Export results to files
    engine.dumpLayerToFile(intersectionResult, "intersection_result.txt", 0);
    engine.dumpLayerToFile(unionResult, "union_result.txt", 0);
    engine.dumpLayerToFile(differenceResult, "difference_result.txt", 0);
    engine.dumpLayerToFile(xorResult, "xor_result.txt", 0);
    std::cout << "Result layers exported to files\n";
    
    std::cout << "\n=== Ground Truth Comparison ===\n";
    
    // Load ground truth layers for comparison (only if we have file layers)
    if (subjectLayer.getPolygonCount() != rectangleLayer.getPolygonCount()) {
        std::cout << "Loading ground truth layers for verification...\n";
        
        // Load ground truth for each operation type
        Layer gtIntersection = engine.createGroundTruthLayer("../testdata/synopsys_simplified_fixed.txt", 
                                                           OperationType::INTERSECTION);
        Layer gtUnion = engine.createGroundTruthLayer("../testdata/synopsys_simplified_fixed.txt", 
                                                     OperationType::UNION);
        Layer gtDifference = engine.createGroundTruthLayer("../testdata/synopsys_simplified_fixed.txt", 
                                                          OperationType::DIFFERENCE);
        Layer gtXor = engine.createGroundTruthLayer("../testdata/synopsys_simplified_fixed.txt", 
                                                   OperationType::XOR);
        
        // Report ground truth layer statistics
        if (!gtIntersection.empty()) {
            std::cout << "Ground truth intersection: " << gtIntersection.getPolygonCount() 
                      << " polygons, " << gtIntersection.getVertexCount() << " vertices\n";
        }
        if (!gtUnion.empty()) {
            std::cout << "Ground truth union: " << gtUnion.getPolygonCount() 
                      << " polygons, " << gtUnion.getVertexCount() << " vertices\n";
        }
        if (!gtDifference.empty()) {
            std::cout << "Ground truth difference: " << gtDifference.getPolygonCount() 
                      << " polygons, " << gtDifference.getVertexCount() << " vertices\n";
        }
        if (!gtXor.empty()) {
            std::cout << "Ground truth XOR: " << gtXor.getPolygonCount() 
                      << " polygons, " << gtXor.getVertexCount() << " vertices\n";
        }
        
        // Create verification comparison visualizations
        std::cout << "Creating verification comparison plots...\n";
        
        if (!gtIntersection.empty()) {
            engine.visualizeVerificationComparison(intersectionResult, gtIntersection, 
                                                  "intersection_verification.png");
        }
        if (!gtUnion.empty()) {
            engine.visualizeVerificationComparison(unionResult, gtUnion, 
                                                  "union_verification.png");
        }
        if (!gtDifference.empty()) {
            engine.visualizeVerificationComparison(differenceResult, gtDifference, 
                                                  "difference_verification.png");
        }
        if (!gtXor.empty()) {
            engine.visualizeVerificationComparison(xorResult, gtXor, 
                                                  "xor_verification.png");
        }
        
        std::cout << "Verification plots created\n";
        
        // Create comprehensive comparison plots
        std::cout << "Creating comprehensive comparison plots...\n";
        
        if (!gtIntersection.empty()) {
            // Simplified polygons comparison
            engine.visualizeComprehensiveComparison(subjectLayer, clipperLayer, intersectionResult, gtIntersection,
                                                   "intersection_comprehensive_simplified.png", false);
            // Raw contours comparison  
            engine.visualizeComprehensiveComparison(subjectLayer, clipperLayer, intersectionResult, gtIntersection,
                                                   "intersection_comprehensive_raw.png", true);
        }
        if (!gtUnion.empty()) {
            engine.visualizeComprehensiveComparison(subjectLayer, clipperLayer, unionResult, gtUnion,
                                                   "union_comprehensive_simplified.png", false);
            engine.visualizeComprehensiveComparison(subjectLayer, clipperLayer, unionResult, gtUnion,
                                                   "union_comprehensive_raw.png", true);
        }
        if (!gtDifference.empty()) {
            engine.visualizeComprehensiveComparison(subjectLayer, clipperLayer, differenceResult, gtDifference,
                                                   "difference_comprehensive_simplified.png", false);
            engine.visualizeComprehensiveComparison(subjectLayer, clipperLayer, differenceResult, gtDifference,
                                                   "difference_comprehensive_raw.png", true);
        }
        if (!gtXor.empty()) {
            engine.visualizeComprehensiveComparison(subjectLayer, clipperLayer, xorResult, gtXor,
                                                   "xor_comprehensive_simplified.png", false);
            engine.visualizeComprehensiveComparison(subjectLayer, clipperLayer, xorResult, gtXor,
                                                   "xor_comprehensive_raw.png", true);
        }
        
        std::cout << "Comprehensive comparison plots created\n";
    } else {
        std::cout << "Using synthetic layers - no ground truth comparison available\n";
    }
    
    std::cout << "\n=== Complex Workflow Example ===\n";
    
    // Example of a more complex workflow
    // Step 1: Create base patterns
    GeometryConfig basePattern;
    basePattern.shape = GeometryConfig::REGULAR_POLYGON;
    basePattern.centerX = 2500;
    basePattern.centerY = 2500;
    basePattern.radius = 300;
    basePattern.numSides = 8;  // Octagon
    
    Layer baseLayer = engine.createLayerFromGeometry(basePattern);
    
    // Step 2: Create masking pattern
    GeometryConfig maskPattern;
    maskPattern.shape = GeometryConfig::RECTANGLE;
    maskPattern.centerX = 2500;
    maskPattern.centerY = 2500;
    maskPattern.width = 400;
    maskPattern.height = 200;
    
    Layer maskLayer = engine.createLayerFromGeometry(maskPattern);
    
    // Step 3: Apply mask (intersection)
    Layer maskedLayer = engine.layerIntersection(baseLayer, maskLayer);
    std::cout << "Step 1 - Masked layer: " << maskedLayer.getPolygonCount() << " polygons\n";
    
    // Step 4: Apply positive offset
    Layer offsetMasked = engine.layerOffset(maskedLayer, 20);
    std::cout << "Step 2 - Offset masked: " << offsetMasked.getPolygonCount() << " polygons\n";
    
    // Step 5: Combine with original base
    Layer finalResult = engine.layerUnion(baseLayer, offsetMasked);
    std::cout << "Step 3 - Final result: " << finalResult.getPolygonCount() << " polygons\n";
    
    // Visualize workflow steps
    engine.visualizeLayer(baseLayer, "workflow_step1_base.png");
    engine.visualizeLayer(maskLayer, "workflow_step2_mask.png");
    engine.visualizeLayer(maskedLayer, "workflow_step3_masked.png");
    engine.visualizeLayer(offsetMasked, "workflow_step4_offset.png");
    engine.visualizeLayer(finalResult, "workflow_step5_final.png");
    
    engine.dumpLayerToFile(finalResult, "workflow_final_result.txt", 0);
    std::cout << "Workflow visualization and results saved\n";
    
    std::cout << "\n=== Performance Statistics ===\n";
    engine.printPerformanceStats();
    
    // Demonstrate layer properties
    std::cout << "\n=== Layer Properties ===\n";
    auto bbox = finalResult.getBoundingBox();
    std::cout << "Final result bounding box: (" << bbox[0] << ", " << bbox[1] 
              << ") to (" << bbox[2] << ", " << bbox[3] << ")\n";
    
    auto gridSize = engine.getCurrentGridSize();
    std::cout << "Engine grid size: " << gridSize[0] << " x " << gridSize[1] << "\n";
    
    std::cout << "\n=== Example Complete ===\n";
    std::cout << "Check output files:\n";
    std::cout << "  - *.png files for basic visualizations\n";
    std::cout << "  - *_verification.png files for ground truth comparisons\n";
    std::cout << "  - *_comprehensive_simplified.png files for comprehensive comparisons (simplified polygons)\n";
    std::cout << "  - *_comprehensive_raw.png files for comprehensive comparisons (raw contours)\n";
    std::cout << "  - *.txt files for polygon data\n";
    
    return 0;
}