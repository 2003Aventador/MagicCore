#!/usr/bin/env python3

################################################################################
# SPMM Performance Analysis Script
#
# Parses msprof output and extracts performance metrics for SPMM kernel
################################################################################

import sys
import pandas as pd
import numpy as np

def parse_op_csv(csv_file):
    """Parse the op_*.csv file from msprof output"""
    try:
        df = pd.read_csv(csv_file)

        # Filter for SPMM kernel operations
        spmm_df = df[df['op_name'].str.contains('spmm', na=False)]

        if spmm_df.empty:
            print("Warning: No SPMM operations found in CSV")
            return None

        # Extract key metrics
        total_time = spmm_df['dur'].sum() / 1e6  # Convert ns to ms
        avg_time = spmm_df['dur'].mean() / 1e6
        min_time = spmm_df['dur'].min() / 1e6
        max_time = spmm_df['dur'].max() / 1e6
        count = len(spmm_df)

        # Calculate FLOPs (A × B)
        # Each operation does: C[M,N] = A[M,K] × B[K,N]
        # But with sparsity, only non-zero blocks are computed
        flops_per_op = 2 * M * N * K  # Dense FLOPs

        # Extract sparsity from kernel parameters if available
        # This is approximate - actual depends on BSR nnz_blocks
        sparsity_factor = 0.1  # Assume 90% sparsity (10% non-zero blocks)
        actual_flops = flops_per_op * sparsity_factor

        perf_metrics = {
            'total_time_ms': total_time,
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'count': count,
            'theoretical_flops': flops_per_op,
            'actual_flops': actual_flops,
            'throughput_gflops': (actual_flops / 1e9) / (avg_time / 1000) if avg_time > 0 else 0
        }

        return perf_metrics

    except Exception as e:
        print(f"Error parsing CSV file: {e}")
        return None

def print_performance_summary(metrics, M, N, K, lda, ldb, ldc, block_size):
    """Print performance summary in a formatted way"""
    if not metrics:
        return

    print("\n" + "=" * 80)
    print("SPMM Performance Summary")
    print("=" * 80)
    print(f"Matrix Dimensions: M={M}, N={N}, K={K}")
    print(f"Leading Dimensions: lda={lda}, ldb={ldb}, ldc={ldc}")
    print(f"BSR Block Size: {block_size}")
    print("-" * 80)
    print(f"Operation Count: {metrics['count']}")
    print(f"Total Time: {metrics['total_time_ms']:.3f} ms")
    print(f"Average Time: {metrics['avg_time_ms']:.3f} ms")
    print(f"Min Time: {metrics['min_time_ms']:.3f} ms")
    print(f"Max Time: {metrics['max_time_ms']:.3f} ms")
    print("-" * 80)
    print(f"Theoretical FLOPs (dense): {metrics['theoretical_flops'] / 1e9:.3f} GFLOPs")
    print(f"Actual FLOPs (sparse): {metrics['actual_flops'] / 1e9:.3f} GFLOPs")
    print(f"Effective Throughput: {metrics['throughput_gflops']:.3f} GFLOPS")
    print("=" * 80)

def main():
    if len(sys.argv) < 7:
        print("Usage: python3 prof.py <op_csv_file> <M> <N> <K> <lda> <ldb> <ldc> <block_size>")
        sys.exit(1)

    csv_file = sys.argv[1]
    M = int(sys.argv[2])
    N = int(sys.argv[3])
    K = int(sys.argv[4])
    lda = int(sys.argv[5])
    ldb = int(sys.argv[6])
    ldc = int(sys.argv[7])
    block_size = int(sys.argv[8])

    print(f"Analyzing performance data from: {csv_file}")
    metrics = parse_op_csv(csv_file)

    if metrics:
        print_performance_summary(metrics, M, N, K, lda, ldb, ldc, block_size)
    else:
        print("Failed to parse performance data")
        sys.exit(1)

if __name__ == "__main__":
    main()
