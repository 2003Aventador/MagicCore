#!/bin/bash

################################################################################
# SPMM Build Helper Script for Ascend 910B3
# Convenient script to clean, compile, and test SPMM with SR-BCRS format
################################################################################

echo "=========================================="
echo "SPMM SR-BCRS Build and Test Helper"
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
    echo "  Testing with small matrix (256×256)..."

    # Generate test data (SR-BCRS format: M N K sparsity d)
    python3 spmm_gen_data.py 256 256 256 85 16 > /dev/null 2>&1

    # Run test
    cd build
    ./spmm 256 256 256 1 0 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ Quick test passed"
    else
        echo "⚠ Quick test failed (check logs with: ./run.sh 256 256 256)"
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
echo "  1. Function test:  ./run.sh 256 256 256"
echo "  2. Performance:    ./run.sh 1024 1024 1024 prof 0"
echo "  3. Custom sparsity: ./run.sh 512 512 512 90 16"
echo ""
echo "See README.md for detailed documentation"
echo "=========================================="