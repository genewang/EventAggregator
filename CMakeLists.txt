cmake_minimum_required(VERSION 3.18)
project(EventAggregator VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture (adjust for your GPUs)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures")

# Build options
option(BUILD_EXAMPLES "Build example programs" ON)
option(ENABLE_NCCL "Enable NCCL for multi-GPU support" OFF)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(include)

# Source files
set(CORE_SOURCES
    src/gpu_aggregator.cu
    src/query_api.cpp
    src/rest_api.cpp
    src/columnar_layout.cpp
    src/simd_query_engine.cpp
)

# Add multi-GPU support if enabled
if(ENABLE_NCCL)
    list(APPEND CORE_SOURCES src/multi_gpu_aggregator.cu)
endif()

# Create library
add_library(event_aggregator ${CORE_SOURCES})
target_link_libraries(event_aggregator 
    PRIVATE CUDA::cudart
    PUBLIC Threads::Threads
)

# NCCL support (optional)
if(ENABLE_NCCL)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(NCCL REQUIRED nccl)
    target_include_directories(event_aggregator PRIVATE ${NCCL_INCLUDE_DIRS})
    target_link_libraries(event_aggregator PRIVATE ${NCCL_LIBRARIES})
    target_compile_definitions(event_aggregator PRIVATE ENABLE_NCCL)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_executable(ingest_demo examples/ingest_demo.cpp)
    target_link_libraries(ingest_demo event_aggregator)
    
    add_executable(query_demo examples/query_demo.cpp)
    target_link_libraries(query_demo event_aggregator)
    
    add_executable(multi_gpu_demo examples/multi_gpu_demo.cpp)
    target_link_libraries(multi_gpu_demo event_aggregator)
    
    # Benchmark suite
    add_executable(benchmark_suite benchmarks/benchmark_suite.cpp)
    target_link_libraries(benchmark_suite event_aggregator)
    target_compile_options(benchmark_suite PRIVATE -mavx2 -mavx512f)  # Enable SIMD
endif()

# Installation
install(TARGETS event_aggregator
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

