#pragma once

#include <string>

namespace gpuLitho {

enum class LayerConstructionMode {
    FILE,
    SYNTHETIC
};

enum class LayerMode {
    SINGLE,
    DUAL
};

enum class OperationType {
    NONE,
    OFFSET,
    INTERSECTION,
    UNION,
    DIFFERENCE,
    XOR,
    MERGE,
    SIZING
};

class OperationTypeUtils {
public:
    // Static mapping function to convert OperationType to string
    static const char* operationTypeToString(OperationType type);
    
    // Convenience function to get operation type as std::string
    static std::string getOperationName(OperationType type);
    
    // Generate filename with operation type prefix
    static std::string generateOperationFilename(OperationType type, const std::string& suffix, const std::string& extension = ".png");
};

} // namespace gpuLitho
