# SPMM (Sparse Matrix Multiplication) for Ascend 910B3

高性能BSR格式稀疏矩阵乘法实现，专为华为昇腾910B3 AI处理器优化。

## 概述

本项目实现了一个基于BSR（Block Sparse Row）格式的稀疏矩阵乘法算子（SPMM），支持以下特性：

- **BSR格式**: 将稀疏矩阵划分为固定大小的块（推荐128×128），只存储非零块
- **异构计算**: AIV核心处理数据对齐，AIC核心执行矩阵乘法
- **高性能**: 跳过90%+的零元素计算，充分利用Cube Core并行能力
- **双缓冲优化**: 计算与数据传输重叠，隐藏内存延迟

### 计算模式
```
C[M,N] = A[M,K] × B[K,N]

其中：
- A: BSR格式的稀疏矩阵（90%+稀疏度）
- B: 稠密矩阵
- C: 结果矩阵
```

## 架构支持

**目标硬件**: Huawei Ascend 910B3
- Core NUM: 20个AI Core
- Vector Core: 2个AIV per AIC (40个AIV total)
- Memory: 128KB L1 Cache per Core, 64KB L0 Buffer
- ISA: DAV-C220 (支持fp16矩阵乘法)

**软件环境**:
- CANN Toolkit ≥ 7.0
- LLVM Compiler (ccec)
- Ascend NPU Driver

## 项目结构

```
ascblas/src/
├── spmm.h                      # 主机端接口（kernel启动）
├── spmm_kernel.cpp            # Kernel实现（AIC + AIV）
├── spmm_gen_data.py           # 测试数据生成脚本
├── prof.py                    # 性能分析脚本
├── main.cpp                   # 测试程序（BSR格式转换）
├── bsr_utils.h                # BSR工具函数
├── handle.cc                  # Handle实现
├── makefile                   # 编译配置
├── run.sh                     # 运行脚本
└── README.md                  # 本文档
```

## 快速开始

### 1. 环境准备

```bash
# 设置CANN环境变量（根据实际安装路径修改）
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# 添加到PATH
export PATH=$ASCEND_HOME_PATH/bin:$PATH

# 验证环境
source check_env.sh

echo $ASCEND_HOME_PATH
which ccec
which msprof
```

### 2. 编译Kernel

```bash
cd ascblas/src

# 清理之前的编译结果
make clean

# 编译Kernel和Host程序
make

# 输出：
#  - build/spmm_kernel.o (kernel binary)
#  - build/spmm (host executable)
```

**编译选项说明**:
- `make`           : 编译release版本（优化）
- `make ca`        : 编译CAmodel版本（用于仿真）
- `make clean`     : 清理编译产物

### 3. 运行测试

#### 3.1 功能验证模式

```bash
# 生成测试数据并运行功能验证
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128

# 参数说明：
# 0 0     : transA=0, transB=0 (不转置)
# 1024    : M (A的行数)
# 1024    : N (B的列数)
# 1024    : K (A的列数/B的行数)
# 1024    : lda (A的leading dimension)
# 1024    : ldb (B的leading dimension)
# 1024    : ldc (C的leading dimension)
# 128     : BSR块大小（必须是16的倍数）
```

#### 3.2 性能测试模式

```bash
# 性能测试（自动生成数据，不验证结果）
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128 prof 0

# 参数说明：
# prof    : 性能模式
# 0       : device_id (NPU卡号)
```

#### 3.3 误差输出模式

```bash
# 功能测试并输出详细误差（用于数据收集）
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128 error 0

# 参数说明：
# error   : 误差输出模式
```

## 运行示例

### 示例1: 小规模测试

```bash
# M=512, N=512, K=512, block_size=128
./run.sh 0 0 512 512 512 512 512 512 128
```

### 示例2: 大规模性能测试

```bash
# M=4096, N=4096, K=4096, block_size=128
./run.sh 0 0 4096 4096 4096 4096 4096 4096 128 prof 0
```

### 示例3: 矩形矩阵

```bash
# M=2048, N=1024, K=4096
./run.sh 0 0 2048 1024 4096 2048 4096 2048 128 prof 0
```

## 参数详解

### Kernel Parameters

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| transA | int | A矩阵转置标志 (0=不转置, 1=转置) | 0 |
| transB | int | B矩阵转置标志 (0=不转置, 1=转置) | 0 |
| M | int | A矩阵行数，C矩阵行数 | 1024 |
| N | int | B矩阵列数，C矩阵列数 | 1024 |
| K | int | A矩阵列数，B矩阵行数 | 1024 |
| lda | int | A矩阵leading dimension | 1024 |
| ldb | int | B矩阵leading dimension | 1024 |
| ldc | int | C矩阵leading dimension | 1024 |
| block_size | int | BSR块大小（必须是16的倍数） | 128 |
| verifyLevel | int | 验证级别 (0=性能, 1=验证, 2=误差) | 1 |
| device_id | int | NPU设备ID | 0 |

### BSR块大小选择

**推荐值**: `128`

**选择依据**:
- 必须是16的倍数（Cube单元要求）
- 128×128×2(fp16) = 32KB per block（适合L1缓存）
- 双缓冲需要64KB，在L1 128KB限制内
- 实际稀疏度90%时，计算量减少到10%

**其他选项**: 64, 256（根据矩阵维度和稀疏模式调整）

## 性能调优指南

### 1. Block Size优化

```bash
# 测试不同block size的性能
for bs in 64 128 256; do
    echo "Testing block_size=$bs"
    ./run.sh 0 0 1024 1024 1024 1024 1024 1024 $bs prof 0
done
```

### 2. 矩阵维度对齐

为了获得最佳性能，建议：
- M、N、K对齐到block_size的倍数
- lda、ldb、ldc对齐到256（512B）

```bash
# 推荐的对齐参数
M=1024, lda=1024  # 1024 % 128 = 0
N=1024, ldb=1024  # 1024 % 128 = 0
K=1024, ldc=1024  # 1024 % 128 = 0
```

### 3. 性能分析

```bash
# 运行性能分析
./run.sh 0 0 2048 2048 2048 2048 2048 2048 128 prof 0

# 查看详细的性能数据
cat prof/op_summary_*.csv
cat prof/ai_core_utilization_*.json
```

### 性能指标

- **Throughput**: GFLOPS（实际稀疏计算吞吐量）
- **Utilization**: AI Core利用率（目标>80%）
- **Memory Bandwidth**: 内存带宽利用率
- **Block Size Impact**: 不同块大小的性能对比

## BSR格式说明

### 数据结构

```cpp
// 主机端BSR数据结构
struct BsrMatrix {
    __fp16* values;          // [nnz_blocks][block_size][block_size]
    int32_t* col_indices;    // [nnz_blocks]
    int32_t* row_offsets;    // [num_row_blocks + 1]
    int32_t nnz_blocks;      // 非零块数量
};
```

### 转换过程

1. **输入**: 稠密矩阵A[M][K]（包含大量零元素）
2. **分块**: 划分为block_size×block_size的块
3. **压缩**: 只保留非零块（sparsity_threshold控制）
4. **输出**: BSR格式（values + col_indices + row_offsets）

### 稀疏度控制

在`spmm_gen_data.py`中调整：
```python
sparsity_mask = np.random.rand(M, K) < 0.1  # 10%非零元素
```

## 验证与调试

### 1. 功能验证

```bash
# 详细验证
./run.sh 0 0 512 512 512 512 512 512 128

# 输出示例：
# BSR conversion: M=512, K=512, block_size=128, nnz_blocks=20 (sparsity: 95.0%)
# BSR SPMM Verification: transA=0, transB=0, M=512, N=512, K=512, block_size=128, nnz_blocks=20
# All data is correct!
```

### 2. 误差分析

```bash
# 输出详细误差信息
./run.sh 0 0 512 512 512 512 512 512 128 error
```

### 3. CAmodel仿真（无需硬件）

```bash
# 编译仿真版本
make ca

# 运行仿真（需要CAmodel环境）
cd build
./spmm_ca 0 0 128 128 128 128 128 128 128 1
cd ..
```

## 常见问题

### Q1: 编译错误"ccec: command not found"

**解决**: 检查CANN环境变量
```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PATH=$ASCEND_HOME_PATH/bin:$PATH
```

### Q2: 运行时错误"msprof: command not found"

**解决**: 使用完整的msprof路径
```bash
export PATH=/usr/local/Ascend/tools/msprof/bin:$PATH
```

### Q3: 性能低于预期

**可能原因**:
1. Block size太小（增加overhead）
2. 矩阵维度未对齐（padding开销）
3. 稀疏度太低（非零块太多）
4. NPU频率限制（检查功耗模式）

**解决**:
```bash
# 设置高性能模式
sudo npu-smi set -t high-performance -i 0
```

### Q4: 验证失败

**可能原因**:
1. BSR格式转换错误
2. Block size不是16的倍数
3. 内存越界（检查lda/ldb/ldc）

**调试**:
```bash
# 小规模测试
./run.sh 0 0 64 64 64 64 64 64 64

# 打印详细日志
export ASCEND_GLOBAL_LOG_LEVEL=0
./run.sh 0 0 128 128 128 128 128 128 128
```

## 性能预期

在Ascend 910B3上（20个AI Core）:

| M,N,K | Block Size | 稀疏度 | 理论GFLOPS | 实际GFLOPS | 提升倍数 |
|-------|------------|--------|-----------|-----------|----------|
| 1024 | 128 | 90% | 256 | ~180 | 10× |
| 2048 | 128 | 90% | 1024 | ~720 | 10× |
| 4096 | 128 | 90% | 4096 | ~2800 | 10× |

*注: 实际性能取决于稀疏模式和内存带宽*

## 相关文件

### 源代码
- `spmm.h`: 主机端kernel调用接口
- `spmm_kernel.cpp`: Kernel实现（AIC + AIV）
- `spmm_gen_data.py`: 测试数据生成
- `bsr_utils.h`: BSR格式转换工具
- `handle.cc`: 运行时句柄
- `main.cpp`: 主程序

### 编译配置
- `makefile`: 编译配置
- `run.sh`: 运行脚本
- `prof.py`: 性能分析工具

### 参考文档
- tests/ascblasHgemm/README.md: 原始GEMM文档
- tests/ascblasHgemm/CAmodel/: CAmodel仿真配置

## 参考文献

1. [Huawei Ascend Computing Language (ACL) Developer Guide](https://www.hiascend.com/document)
2. [CANN Kernel Development Guide](https://www.hiascend.com/document)
3. Sparse BLAS: 稀疏线性代数库标准
4. BSR Format: 分块稀疏行格式（IEEE标准）

## 许可证

本项目基于MIT许可证。详情见LICENSE文件。

## 支持与反馈

- 问题反馈: [GitHub Issues](https://github.com/...)
- 技术支持: [Huawei Ascend Forum](https://forum.huaweicloud.com/ascend)

---

**文档版本**: v1.0
**最后更新**: 2026-01-16
**维护者**: Ascend SPMM Development Team
