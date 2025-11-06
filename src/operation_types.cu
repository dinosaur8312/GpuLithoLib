#include "../include/operation_types.h"

namespace gpuLitho {

const char* OperationTypeUtils::operationTypeToString(OperationType type) {
    switch (type) {
        case OperationType::NONE: return "NONE";
        case OperationType::OFFSET: return "OFFSET";
        case OperationType::INTERSECTION: return "INTERSECTION";
        case OperationType::UNION: return "UNION";
        case OperationType::DIFFERENCE: return "DIFFERENCE";
        case OperationType::XOR: return "XOR";
        case OperationType::MERGE: return "MERGE";
        case OperationType::SIZING: return "SIZING";
        default: return "UNKNOWN";
    }
}

std::string OperationTypeUtils::getOperationName(OperationType type) {
    return std::string(operationTypeToString(type));
}

std::string OperationTypeUtils::generateOperationFilename(OperationType type, const std::string& suffix, const std::string& extension) {
    return getOperationName(type) + "_" + suffix + extension;
}

} // namespace gpuLitho