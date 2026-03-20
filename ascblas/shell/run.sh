#!/bin/bash

################################################################################
# SPMM (Sparse Matrix Multiplication) Run Script for Ascend 910B3
#
# This script compiles and runs the BSR-format SPMM kernel on Ascend 910B3
# Usage: ./run.sh <transA> <transB> <M> <N> <K> <lda> <ldb> <ldc> <block_size> [mode] [device_id]
#
# Parameters:
#   transA, transB: 0 (no transpose) or 1 (transpose)
#   M, N, K: Matrix dimensions for C = A × B
#   lda, ldb, ldc: Leading dimensions
#   block_size: BSR block size (multiple of 16, recommended: 128)
#   mode: "prof" (performance test), "error" (functional test with error output)
#         or default (functional test with verification)
#   device_id: NPU device ID (default: 0)
################################################################################

set -e

# Error handling
function check_error {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed!"
        exit 1
    fi
}

# Set Canndev DLL path for 910B3
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

OP_NAME="spmm_kernel"
EXECUTABLE="spmm"

# Parse command line arguments
if [ $# -lt 9 ]; then
    echo "Usage: $0 <transA> <transB> <M> <N> <K> <lda> <ldb> <ldc> <block_size> [mode] [device_id]"
    echo ""
    echo "Example:"
    echo "  $0 0 0 1024 1024 1024 1024 1024 1024 128"
    echo "  $0 0 0 1024 1024 1024 1024 1024 1024 128 prof 0"
    echo "  $0 0 0 1024 1024 1024 1024 1024 1024 128 error 0"
    exit 1
fi

transA=$1
transB=$2
M=$3
N=$4
K=$5
lda=$6
ldb=$7
ldc=$8
block_size=$9

# Set mode (prof/error/verify)
MODE="verify"
if [ $# -ge 10 ]; then
    if [ "${10}" == "prof" ]; then
        MODE="prof"
    elif [ "${10}" == "error" ]; then
        MODE="error"
    else
        MODE="verify"
    fi
fi

# Set device ID
device_id=0
if [ $# -ge 11 ]; then
    device_id=${11}
fi

# Set verify level based on mode
verifyLevel=1  # Default: verify correctness
if [ "$MODE" == "prof" ]; then
    verifyLevel=0  # Performance mode: no verification
elif [ "$MODE" == "error" ]; then
    verifyLevel=2  # Output error for CSV collection
fi

echo "=========================================="
echo "SPMM Configuration:"
echo "  Operation: C = A × B"
echo "  Matrix dimensions: M=${M}, N=${N}, K=${K}"
echo "  Leading dims: lda=${lda}, ldb=${ldb}, ldc=${ldc}"
echo "  BSR block size: ${block_size}"
echo "  Transpose: transA=${transA}, transB=${transB}"
echo "  Mode: ${MODE}"
echo "  Device ID: ${device_id}"
echo "=========================================="

# Create directories
mkdir -p data
mkdir -p build
mkdir -p prof

# Step 1: Generate test data (skip for performance mode)
if [ "$MODE" != "prof" ]; then
    echo "[1/4] Generating test data..."
    python3 ${EXECUTABLE}_gen_data.py $transA $transB $M $N $K $lda $ldb $ldc $block_size
    check_error "Data generation"
else
    echo "[1/4] Performance mode: Skipping data generation"
fi

# Step 2: Compile kernel and host code
if [ "$MODE" == "verify" ]; then
    echo "[2/4] Compiling SPMM kernel and host code..."
    make clean > /dev/null
    make > /dev/null 2>&1
    check_error "Compilation"
else
    echo "[2/4] Compiling SPMM kernel and host code..."
    make clean > /dev/null
    make > /dev/null 2>&1
    check_error "Compilation"
fi

# Step 3: Run the executable
echo "[3/4] Running SPMM kernel..."
cd build

if [ "$MODE" == "prof" ]; then
    # Performance profiling mode
    echo "  Running performance analysis with msprof..."
    rm -rf ../prof/*

    # Enable performance profiling
    export ASCEND_GLOBAL_LOG_LEVEL=1
    export ASCEND_SLOG_PRINT_TO_STDOUT=0

    msprof --application="./${EXECUTABLE} $transA $transB $M $N $K $lda $ldb $ldc $block_size $verifyLevel $device_id" \
           --output=../prof \
           --ai-core=on \
           --ai-cpu=off \
           --task-time=on \
           --task-trace=on
    check_error "Performance profiling"

    # Process profiling results
    echo "[4/4] Processing profiling results..."
    cd ..
    if [ -f "prof.py" ]; then
        python3 prof.py $(find prof -name "op_*.csv" 2>/dev/null | head -1) $M $N $K $lda $ldb $ldc $block_size
    else
        echo "  Profiling completed. Results in prof/ directory"
        ls -lh prof/
    fi
else
    # Functional verification mode
    echo "  Running functional verification..."
    ./${EXECUTABLE} $transA $transB $M $N $K $lda $ldb $ldc $block_size $verifyLevel $device_id
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
