cmake_minimum_required(VERSION 3.16)

# Option to force a specific platform
set(GPU_BACKEND "AUTO" CACHE STRING "Choose GPU backend: AUTO, CUDA, or HIP")
set_property(CACHE GPU_BACKEND PROPERTY STRINGS AUTO CUDA HIP)

# Set ROCm/HIP path if available
if(EXISTS "/opt/rocm")
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
elseif(EXISTS "/opt/rocm-6.2.2")
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm-6.2.2)
endif()

# Auto-detect GPU platform if not specified
if(GPU_BACKEND STREQUAL "AUTO")
    # Check for nvcc (CUDA compiler) first
    find_program(NVCC_EXECUTABLE nvcc)
    if(NVCC_EXECUTABLE)
        message(STATUS "CUDA found - using CUDA backend")
        set(USE_CUDA ON)
    else()
        # Check for hipcc (HIP compiler)
        find_program(HIPCC_EXECUTABLE hipcc)
        if(HIPCC_EXECUTABLE)
            message(STATUS "HIP found - using HIP backend")
            set(USE_HIP ON)
        else()
            message(FATAL_ERROR "No GPU backend found. Please install CUDA or HIP/ROCm")
        endif()
    endif()
elseif(GPU_BACKEND STREQUAL "CUDA")
    set(USE_CUDA ON)
    message(STATUS "Forcing CUDA backend")
elseif(GPU_BACKEND STREQUAL "HIP")
    set(USE_HIP ON)
    message(STATUS "Forcing HIP backend")
endif()

# Set project with appropriate language
if(USE_HIP)
    project(GpuLithoLib LANGUAGES CXX)
    enable_language(HIP)
else()
    project(GpuLithoLib LANGUAGES CXX CUDA)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(LIBRARY_SOURCES
    src/LayerImpl.cu
    src/GpuOperations.cu
    src/IntersectionCompute.cu
    src/ContourProcessing.cu
    src/GpuLithoLib.cu
    src/operation_types.cu
)

# Create static library based on backend
if(USE_HIP)
    # HIP Configuration
    find_package(HIP REQUIRED)

    # Create library
    add_library(GpuLithoLib STATIC ${LIBRARY_SOURCES})

    # Set HIP properties for source files
    set_source_files_properties(${LIBRARY_SOURCES} PROPERTIES LANGUAGE HIP)

    # Set HIP architectures
    set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a;gfx1030;gfx1100" CACHE STRING "HIP architectures")

    # HIP compile flags
    target_compile_options(GpuLithoLib PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:
            --offload-arch=gfx906
            --offload-arch=gfx908
            --offload-arch=gfx90a
            --offload-arch=gfx1030
            --offload-arch=gfx1100
        >
    )

    # HIP definitions
    target_compile_definitions(GpuLithoLib PRIVATE __HIP_PLATFORM_AMD__)

    # Link HIP runtime
    target_link_libraries(GpuLithoLib PRIVATE
        hip::device
        ${OpenCV_LIBS}
    )

else()
    # CUDA Configuration
    find_package(CUDAToolkit REQUIRED)

    # Create library
    add_library(GpuLithoLib STATIC ${LIBRARY_SOURCES})

    # Set CUDA properties
    set_property(TARGET GpuLithoLib PROPERTY CUDA_STANDARD 17)
    set_property(TARGET GpuLithoLib PROPERTY CUDA_STANDARD_REQUIRED ON)
    set_property(TARGET GpuLithoLib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    # Set CUDA architectures
    set_property(TARGET GpuLithoLib PROPERTY CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89;90")

    # CUDA compile flags
    target_compile_options(GpuLithoLib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    )

    # Link CUDA libraries
    target_link_libraries(GpuLithoLib PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
        ${OpenCV_LIBS}
    )
endif()

# Example executables
if(USE_HIP)
    # HIP executables
    add_executable(example_usage examples/example_usage.cu)
    set_source_files_properties(examples/example_usage.cu PROPERTIES LANGUAGE HIP)
    target_compile_definitions(example_usage PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(example_usage GpuLithoLib)

    add_executable(test_intersection examples/test_intersection.cu)
    set_source_files_properties(examples/test_intersection.cu PROPERTIES LANGUAGE HIP)
    target_compile_definitions(test_intersection PRIVATE __HIP_PLATFORM_AMD__)
    target_link_libraries(test_intersection GpuLithoLib)
else()
    # CUDA executables
    add_executable(example_usage examples/example_usage.cu)
    set_property(TARGET example_usage PROPERTY CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89;90")
    target_link_libraries(example_usage GpuLithoLib)

    add_executable(test_intersection examples/test_intersection.cu)
    set_property(TARGET test_intersection PROPERTY CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89;90")
    target_link_libraries(test_intersection GpuLithoLib)
endif()

# Copy operation types source file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/operation_types.h ${CMAKE_CURRENT_BINARY_DIR}/operation_types.h COPYONLY)

# Installation
install(TARGETS GpuLithoLib
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

install(FILES include/GpuLithoLib.h include/gpuLitho.h include/operation_types.h
        DESTINATION include)

# Print configuration summary
message(STATUS "=====================================")
message(STATUS "GpuLithoLib Build Configuration:")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
if(USE_HIP)
    message(STATUS "  GPU Backend: HIP/ROCm")
    message(STATUS "  HIP Version: ${HIP_VERSION}")
    message(STATUS "  Target Architectures: ${CMAKE_HIP_ARCHITECTURES}")
else()
    message(STATUS "  GPU Backend: CUDA")
    message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
    get_property(CUDA_ARCHS TARGET GpuLithoLib PROPERTY CUDA_ARCHITECTURES)
    message(STATUS "  Target Architectures: ${CUDA_ARCHS}")
endif()
message(STATUS "  OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "=====================================")
