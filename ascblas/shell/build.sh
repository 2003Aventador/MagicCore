#!/bin/bash

################################################################################
# SPMM Build Helper Script
# Convenient script to clean, compile, and test SPMM on Ascend 910B3
################################################################################

echo "=========================================="
echo "SPMM Build and Test Helper"
echo "=========================================="

# Check environment
echo "[1/4] Checking environment..."
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "Error: ASCEND_HOME_PATH is not set!"
    echo "Please source the CANN environment:"
    echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    exit 1
fi

echo "  CANN Path: $ASCEND_HOME_PATH"
echo "✓ Environment check passed"

# Clean previous build
echo "[2/4] Cleaning previous build..."
make clean > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Clean completed"
else
    echo "  Clean skipped (no previous build)"
fi

# Compile
echo "[3/4] Compiling SPMM..."
make > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Compilation successful"
else
    echo "✗ Compilation failed!"
    echo "  Run 'make' to see detailed errors"
    exit 1
fi

# Quick test
echo "[4/4] Running quick test..."
if [ -f "./build/spmm" ]; then
    echo "  Executable created: build/spmm"
    echo "  Testing with small matrix (64×64)..."

    # Generate test data
    python3 spmm_gen_data.py 0 0 64 64 64 64 64 64 64 > /dev/null 2>&1

    # Run test
    cd build
    ./spmm 0 0 64 64 64 64 64 64 64 1 0 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ Quick test passed"
    else
        echo "⚠ Quick test failed (check logs with: ./run.sh 0 0 64 64 64 64 64 64 64)"
    fi
    cd ..
else
    echo "✗ Executable not found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "Quick start:"
echo "  1. Function test:  ./run.sh 0 0 512 512 512 512 512 512 128"
echo "  2. Performance:    ./run.sh 0 0 1024 1024 1024 1024 1024 1024 128 prof 0"
echo "  3. Detailed test:  ./run.sh 0 0 2048 2048 2048 2048 2048 2048 128 error 0"
echo ""
echo "See README.md for detailed documentation"
echo "=========================================="
