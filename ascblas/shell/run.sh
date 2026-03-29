#!/bin/bash

################################################################################
# SPMM (Sparse Matrix Multiplication) Run Script for Ascend 910B3 - SR-BCRS Format
#
# Usage: ./run.sh <M> <N> <K> [sparsity] [d] [mode] [device_id]
#
# Parameters:
#   M, N, K: Matrix dimensions for C = A × B
#   sparsity: Sparsity percentage for A matrix (default: 85)
#   d: Vector block dimension (default: 16, should match vec_length in kernel)
#   mode: "prof" (performance test) or default (functional test)
#   device_id: NPU device ID (default: 0)
#
# Examples:
#   ./run.sh 256 256 256                    # Basic test with default sparsity
#   ./run.sh 512 512 512 90                 # 90% sparsity
#   ./run.sh 1024 1024 1024 85 16 prof 0    # Performance test
################################################################################

set -e

# Error handling
function check_error {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed!"
        exit 1
    fi
}

# Set library path for Ascend
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# 这里和spmm.h的编译绑定是对应的
OP_NAME="spmm_kernel"
EXECUTABLE="spmm"

# Parse command line arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <M> <N> <K> [sparsity] [d] [mode] [device_id]"
    echo ""
    echo "Parameters:"
    echo "  M, N, K:  Matrix dimensions"
    echo "  sparsity: Sparsity percentage for A (default: 85)"
    echo "  d:        Vector block dimension (default: 16)"
    echo "  mode:     prof (performance) or default (functional)"
    echo "  device_id: NPU device ID (default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0 256 256 256"
    echo "  $0 512 512 512 90"
    echo "  $0 1024 1024 1024 85 16 prof 0"
    exit 1
fi

M=$1
N=$2
K=$3
sparsity=${4:-85}
d=${5:-16}

# Set mode (prof/default)
MODE="verify"
if [ $# -ge 6 ]; then
    if [ "${6}" == "prof" ]; then
        MODE="prof"
    else
        MODE="verify"
    fi
fi

# Set device ID
device_id=0
if [ $# -ge 7 ]; then
    device_id=${7}
fi

# Set verify level based on mode
verifyLevel=1  # Default: verify correctness
if [ "$MODE" == "prof" ]; then
    verifyLevel=0  # Performance mode: no verification
fi

echo "=========================================="
echo "SPMM SR-BCRS Configuration:"
echo "  Operation: C = A × B"
echo "  Matrix dimensions: M=${M}, N=${N}, K=${K}"
echo "  Sparsity: ${sparsity}%"
echo "  Vector block dimension: d=${d}"
echo "  Mode: ${MODE}"
echo "  Device ID: ${device_id}"
echo "=========================================="

# Create directories
mkdir -p ../data
mkdir -p build
mkdir -p prof

# Step 1: Generate test data
echo "[1/3] Generating SR-BCRS test data..."
python3 spmm_gen_data.py ${M} ${N} ${K} ${sparsity} ${d}
check_error "Data generation"

# Step 2: Compile kernel and host code
echo "[2/3] Compiling SPMM kernel and host code..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1
check_error "Compilation"

# Step 3: Run the executable
echo "[3/3] Running SPMM kernel..."
cd build

if [ "$MODE" == "prof" ]; then
    # Performance profiling mode
    echo "  Running performance analysis with msprof..."
    rm -rf ../prof/*

    # Enable performance profiling
    export ASCEND_GLOBAL_LOG_LEVEL=1
    export ASCEND_SLOG_PRINT_TO_STDOUT=0

    msprof --application="./${EXECUTABLE} ${M} ${N} ${K} ${verifyLevel} ${device_id}" \
           --output=../prof \
           --ai-core=on \
           --ai-cpu=off \
           --task-time=on \
           --task-trace=on
    check_error "Performance profiling"

    # Process profiling results
    echo ""
    cd ..
    echo "Profiling completed. Results in prof/ directory"
    ls -lh prof/
else
    # Functional verification mode
    echo "  Running functional verification..."
    ./${EXECUTABLE} ${M} ${N} ${K} ${verifyLevel} ${device_id}
    check_error "Kernel execution"

    cd ..
fi

echo ""
echo "=========================================="
echo "SPMM execution completed successfully!"
echo "=========================================="

# Optional: Show profiling summary
if [ "$MODE" == "prof" ]; then
    echo ""
    echo "Performance Summary:"
    if [ -d "prof" ]; then
        echo "  Profiling data: prof/"
        ls prof/*.json 2>/dev/null | head -5 | while read -r file; do
            echo "    - $file"
        done
    fi
fi