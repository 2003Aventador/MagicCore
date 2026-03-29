#!/usr/bin/env python3

################################################################################
# SPMM Performance Analysis Script for SR-BCRS Format
#
# Parses msprof output and extracts performance metrics for SPMM kernel
################################################################################

import sys
import os

def parse_op_csv(csv_file):
    """Parse the op_*.csv file from msprof output"""
    try:
        import pandas as pd
        
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

        perf_metrics = {
            'total_time_ms': total_time,
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'count': count
        }

        return perf_metrics

    except ImportError:
        print("Warning: pandas not installed, using basic parsing")
        return None
    except Exception as e:
        print(f"Error parsing CSV file: {e}")
        return None

def calculate_theoretical_flops(M, N, K, sparsity, d):
    """
    计算理论FLOPs
    - 稀疏矩阵A的非零元素比例: (100-sparsity)/100
    - 每个非零元素是d×1向量，实际计算d×N
    """
    density = (100 - sparsity) / 100.0
    nnz_elements = M * K * density  # 非零元素个数
    flops_per_element = 2 * d * N   # 每个非零向量块的计算量
    total_flops = nnz_elements * flops_per_element
    return total_flops

def print_performance_summary(metrics, M, N, K, sparsity, d):
    """Print performance summary in a formatted way"""
    print("\n" + "=" * 80)
    print("SPMM SR-BCRS Performance Summary")
    print("=" * 80)
    print(f"Matrix Dimensions: M={M}, N={N}, K={K}")
    print(f"Sparsity: {sparsity}%")
    print(f"Vector Block Dimension: d={d}")
    print("-" * 80)
    
    if metrics:
        print(f"Operation Count: {metrics['count']}")
        print(f"Total Time: {metrics['total_time_ms']:.3f} ms")
        print(f"Average Time: {metrics['avg_time_ms']:.3f} ms")
        print(f"Min Time: {metrics['min_time_ms']:.3f} ms")
        print(f"Max Time: {metrics['max_time_ms']:.3f} ms")
        print("-" * 80)
        
        # 计算理论性能
        theoretical_flops = calculate_theoretical_flops(M, N, K, sparsity, d)
        if metrics['avg_time_ms'] > 0:
            throughput_gflops = (theoretical_flops / 1e9) / (metrics['avg_time_ms'] / 1000)
            print(f"Theoretical FLOPs: {theoretical_flops / 1e9:.3f} GFLOPs")
            print(f"Effective Throughput: {throughput_gflops:.3f} GFLOPS")
    else:
        print("Performance metrics not available (pandas not installed)")
    
    print("=" * 80)

def main():
    if len(sys.argv) < 6:
        print("Usage: python3 prof.py <op_csv_file> <M> <N> <K> <sparsity> <d>")
        print("  op_csv_file: Path to msprof op_*.csv file")
        print("  M, N, K: Matrix dimensions")
        print("  sparsity: Sparsity percentage")
        print("  d: Vector block dimension")
        print("\nExample:")
        print("  python3 prof.py prof/op_summary_*.csv 1024 1024 1024 85 16")
        sys.exit(1)

    csv_file = sys.argv[1]
    M = int(sys.argv[2])
    N = int(sys.argv[3])
    K = int(sys.argv[4])
    sparsity = int(sys.argv[5])
    d = int(sys.argv[6])

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        sys.exit(1)

    print(f"Analyzing performance data from: {csv_file}")
    metrics = parse_op_csv(csv_file)

    print_performance_summary(metrics, M, N, K, sparsity, d)

if __name__ == "__main__":
    main()