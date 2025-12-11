#!/bin/bash
# Build script for Event Aggregator

set -e

BUILD_DIR="build"
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"

# Parse arguments
ENABLE_NCCL=false
CUDA_ARCH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --nccl)
            ENABLE_NCCL=true
            shift
            ;;
        --cuda-arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--nccl] [--cuda-arch ARCH]"
            echo "  --nccl          Enable NCCL for multi-GPU support"
            echo "  --cuda-arch      CUDA architecture (e.g., '75;80;86')"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set CUDA architecture if provided
if [ -n "$CUDA_ARCH" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
fi

# Enable NCCL if requested
if [ "$ENABLE_NCCL" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_NCCL=ON"
fi

echo "Building Event Aggregator..."
echo "CMake args: $CMAKE_ARGS"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. $CMAKE_ARGS
cmake --build . -j$(nproc)

echo ""
echo "Build complete! Examples are in: $BUILD_DIR/examples/"
echo ""
echo "To run examples:"
echo "  ./build/examples/ingest_demo"
echo "  ./build/examples/query_demo"

