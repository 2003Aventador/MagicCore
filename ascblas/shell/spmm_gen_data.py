#!/usr/bin/env python3
################################################################################
# SPMM (Sparse Matrix Multiplication) Data Generator
#
# Generates test data for BSR-format SPMM kernel
# Creates: A.bin (dense), B.bin, C.bin
################################################################################

import sys
import os
import numpy as np

def write_bin_file(file_path, data):
    """Write numpy array to binary file"""
    data.tofile(file_path)
    print(f"  Generated: {file_path} ({data.shape}, {data.dtype})")

def generate_spmm_data(transA, transB, M, N, K, lda, ldb, ldc, block_size, sparsity):
    """
    Generate test data for SPMM (C = A * B)
    A: M * K matrix (will be converted to BSR format)
    B: K * N matrix
    C: M * N result matrix
    """
    # 计算非零概率 (Density)
    probability = 1.0 - sparsity

    print("\nGenerating SPMM test data...")
    print(f"  Matrix dimensions: M={M}, N={N}, K={K}")
    print(f"  Leading dims: lda={lda}, ldb={ldb}, ldc={ldc}")
    print(f"  BSR block size: {block_size}")
    print(f"  Target Sparsity: {sparsity} (Probability: {probability:.4f})")

    # Create output directory
    os.makedirs("data", exist_ok=True)

    # Generate matrix A (dense, will be converted to BSR)
    # Use random values with controlled sparsity
    print(f"\n  Generating matrix A (with ~{sparsity*100:.1f}% sparsity)...")
    A_dense = np.random.randn(M, K).astype(np.float16) * 0.1

    # Apply sparsity: randomly zero out elements based on sparsity
    sparsity_mask = np.random.rand(M, K) < probability 
    A_dense = A_dense * sparsity_mask

    # Ensure minimum non-zero blocks for meaningful testing
    # num_row_blocks = (M + block_size - 1) // block_size
    # num_col_blocks = (K + block_size - 1) // block_size
    
    # # Calculate minimum blocks based on density (instead of hardcoded // 10)
    # min_nnz_blocks = max(1, int(num_row_blocks * num_col_blocks * probability)) 

    # nnz_blocks = 0
    # while nnz_blocks < min_nnz_blocks:
    #     for rb in range(num_row_blocks):
    #         for cb in range(num_col_blocks):
    #             # Use the variable probability instead of 0.1
    #             if np.random.rand() < probability: 
    #                 row_start = rb * block_size
    #                 row_end = min(row_start + block_size, M)
    #                 col_start = cb * block_size
    #                 col_end = min(col_start + block_size, K)
                    
    #                 block_values = np.random.randn(row_end - row_start,
    #                                                col_end - col_start).astype(np.float16)
    #                 A_dense[row_start:row_end, col_start:col_end] = block_values
    #                 nnz_blocks += 1
    
    # print(f"    Generated {nnz_blocks} non-zero blocks")

    # Write A in dense format (will be converted to BSR by host code)
    if transA == 0:  # No transpose
        A_padded = np.zeros((lda, K), dtype=np.float16)
        A_padded[:M, :K] = A_dense
        write_bin_file("data/A.bin", A_padded)
    else:  # Transpose
        A_transposed = A_dense.T
        A_padded = np.zeros((lda, M), dtype=np.float16)
        A_padded[:K, :M] = A_transposed
        write_bin_file("data/A.bin", A_padded)

    # Generate matrix B
    print("\n  Generating matrix B...")
    B = np.random.randn(K, N).astype(np.float16) * 0.1
    
    if transB == 0:  # No transpose
        B_padded = np.zeros((ldb, N), dtype=np.float16)
        B_padded[:K, :N] = B
        write_bin_file("data/B.bin", B_padded)
    else:  # Transpose
        B_transposed = B.T
        B_padded = np.zeros((ldb, K), dtype=np.float16)
        B_padded[:N, :K] = B_transposed
        write_bin_file("data/B.bin", B_padded)

    # # Generate initial C matrix (usually zeros)
    # print("\n  Generating matrix C...")
    # C = np.zeros((ldc, N), dtype=np.float16)
    # write_bin_file("data/C.bin", C)

    # Compute expected result for verification
    # print("\n  Computing expected result...")
    # if transA == 0 and transB == 0:
    #     C_expect = A_dense.astype(np.float32) @ B.astype(np.float32)
    # elif transA == 1 and transB == 0:
    #     C_expect = A_dense.T.astype(np.float32) @ B.astype(np.float32)
    # elif transA == 0 and transB == 1:
    #     C_expect = A_dense.astype(np.float32) @ B.T.astype(np.float32)
    # else:  # transA == 1 and transB == 1
    #     C_expect = A_dense.T.astype(np.float32) @ B.T.astype(np.float32)

    # C_expect_padded = np.zeros((ldc, N), dtype=np.float16)
    # C_expect_padded[:M, :N] = C_expect.astype(np.float16)
    # write_bin_file("data/C_expect.bin", C_expect_padded)

    print("\n  Data generation completed successfully!")

if __name__ == "__main__":

    transA = int(sys.argv[1])
    transB = int(sys.argv[2])
    M = int(sys.argv[3])
    N = int(sys.argv[4])
    K = int(sys.argv[5])
    lda = int(sys.argv[6])
    ldb = int(sys.argv[7])
    ldc = int(sys.argv[8])
    block_size = int(sys.argv[9])
    
    # 获取 sparsity 参数，默认 0.9
    sparsity = 0.9

    # 参数检查
    if sparsity < 0.0 or sparsity >= 1.0:
        print(f"Error: Sparsity {sparsity} must be between 0.0 and 1.0")
        sys.exit(1)

    print("=" * 60)
    print("SPMM Data Generator")
    print("=" * 60)

    generate_spmm_data(transA, transB, M, N, K, lda, ldb, ldc, block_size, sparsity)
