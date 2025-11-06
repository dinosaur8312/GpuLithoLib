cmake_minimum_required(VERSION 3.18)
project(GpuLithoLib LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Compiler-specific options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(LIBRARY_SOURCES
    src/LayerImpl.cu
    src/GpuOperations.cu
    src/GpuLithoLib.cu
    src/operation_types.cu
)

# Create static library
add_library(GpuLithoLib STATIC ${LIBRARY_SOURCES})

# Link libraries
target_link_libraries(GpuLithoLib 
    ${OpenCV_LIBS}
    CUDA::cudart
    CUDA::cuda_driver
)

# Set properties for CUDA compilation
set_property(TARGET GpuLithoLib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Example executable
add_executable(example_usage examples/example_usage.cu)
target_link_libraries(example_usage GpuLithoLib)

# Copy operation types source file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/operation_types.h ${CMAKE_CURRENT_BINARY_DIR}/operation_types.h COPYONLY)

# Installation
install(TARGETS GpuLithoLib
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(FILES include/GpuLithoLib.h include/gpuLitho.h include/operation_types.h
        DESTINATION include)

# Print configuration
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")