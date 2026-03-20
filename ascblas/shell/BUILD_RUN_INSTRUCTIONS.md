# SPMM 编译与运行指南（Ascend 910B3）

## 环境要求

### 硬件
- **华为昇腾910B3 AI处理器**
  - 20个AI Core
  - 32GB HBM内存
  - 支持FP16矩阵加速

### 软件
- **操作系统**: Ubuntu 18.04/20.04 或 openEuler 20.03
- **CANN Toolkit**: 7.0或更高版本
- **Python**: 3.7+
- **NumPy**: 1.19+
- **编译器**: GCC 7.5+, LLVM (ccec)

## 首次运行准备

### 1. 设置环境变量

**非常重要**: 必须先设置CANN环境变量！

```bash
# 根据实际安装路径修改
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 执行环境设置脚本
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证环境设置
echo $ASCEND_HOME_PATH
which ccec
which msprof
```

### 2. 快速环境检查

```bash
cd ascblas/src
./check_env.sh
```

**期望输出**:
```
✓ ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
✓ ccec compiler found
✓ g++ compiler found
✓ ld.lld linker found
✓ Found 1 NPU device(s)
✓ NPU Type: 910B3 detected
...
Environment check passed!
```

如果检查失败，请根据提示修复环境问题。

### 3. 快速构建与测试

```bash
./build.sh
```

**输出**:
```
[1/4] Checking environment...
[2/4] Cleaning previous build...
[3/4] Compiling SPMM...
[4/4] Running quick test...
  Executable created: build/spmm
  Testing with small matrix (64×64)...
✓ Quick test passed

Build completed successfully!
```

## 编译指令详解

### 手动编译步骤

#### 步骤1: 进入源码目录

```bash
cd ascblas/src
```

#### 步骤2: 编译Kernel

```bash
# 完整编译（kernel + host）
make clean && make
```

**详细输出**:
```
rm -rf build/
mkdir -p build

# 编译AIC kernel (Cube计算核心)
ccec -std=c++17 -c -x cce -O2 spmm_kernel.cpp --cce-aicore-arch=dav-c220-cube \
     --cce-aicore-only -I... -o build/spmm_kernel_aic.o

# 编译AIV kernel (Vector预处理核心)
ccec -std=c++17 -c -x cce -O2 spmm_kernel.cpp --cce-aicore-arch=dav-c220-vec \
     --cce-aicore-only -I... -o build/spmm_kernel_aiv.o

# 链接AIC和AIV object文件
ld.lld -m aicorelinux -Ttext=0 -z separate-code \
       build/spmm_kernel_aic.o build/spmm_kernel_aiv.o \
       --allow-multiple-definition -static -o build/spmm_kernel.o

# 编译Host程序
g++ -O2 -fpic -std=c++17 handle.cc main.cpp \
    -I$ASCEND_HOME_PATH/include -I../include \
    -L$ASCEND_HOME_PATH/lib64 -o build/spmm \
    -lascendcl -lruntime -lstdc++ -lpthread
```

**验证编译结果**:
```bash
ls -lh build/
total 1.2M
-rwxr-xr-x 1 user group 856K spmm          # Host executable
-rw-r--r-- 1 user group 340K spmm_kernel.o # Kernel binary
-rw-r--r-- 1 user group  24K spmm_kernel_aic.o
-rw-r--r-- 1 user group  24K spmm_kernel_aiv.o
```

#### 步骤3: 检查编译输出

```bash
# 确认kernel名称（重要！）
cd build
strings spmm_kernel.o | grep spmm
```

**应该包含**:
```
spmm_kernel_aic
spmm_kernel_aiv
```

如果找不到，说明kernel名称不匹配，需要检查spmm_kernel.cpp中的函数名。

## 运行指令详解

### 功能验证模式（推荐首次运行）

#### 示例1: 小型矩阵验证

```bash
./run.sh 0 0 128 128 128 128 128 128 128
```

**参数详解**:
- `0`: transA (不转置)
- `0`: transB (不转置)
- `128`: M (A行数/C行数)
- `128`: N (B列数/C列数)
- `128`: K (A列数/B行数)
- `128`: lda (A的leading dimension)
- `128`: ldb (B的leading dimension)
- `128`: ldc (C的leading dimension)
- `128`: block_size (BSR块大小)

**期望输出**:
```
==========================================
SPMM Configuration:
  Operation: C = A × B
  Matrix dimensions: M=128, N=128, K=128
  Leading dims: lda=128, ldb=128, ldc=128
  BSR block size: 128
  Transpose: transA=0, transB=0
  Mode: verify
  Device ID: 0
==========================================
[1/4] Generating test data...
  Generated: data/A.bin ((128, 128), float16)
  Generated: data/B.bin ((128, 128), float16)
  Generated: data/C.bin ((128, 128), float16)
  Computing expected result...
  Generated: data/C_expect.bin ((128, 128), float16)
  Data generation completed successfully!
BSR conversion: M=128, K=128, block_size=128, nnz_blocks=16 (sparsity: 87.50%)
[2/4] Compiling SPMM kernel and host code...
[3/4] Running SPMM kernel...
BSR SPMM Verification: transA=0, transB=0, M=128, N=128, K=512, block_size=128, nnz_blocks=16
All data is correct!
[4/4] Processing profiling results...

==========================================
SPMM execution completed successfully!
==========================================
```

#### 示例2: 中等规模验证

```bash
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128
```

**关键点**:
- 耗时约1-2秒
- 自动检测稀疏度（~90%）
- 验证结果正确性

### 性能测试模式

#### 示例3: 性能基准测试

```bash
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128 prof 0
```

**特点**:
- `prof`: 性能模式（不验证结果）
- `0`: device_id (NPU卡号)
- 自动生成数据，不保存中间结果
- 使用msprof进行性能分析

**期望输出**:
```
==========================================
SPMM Configuration:
  Operation: C = A × B
  Matrix dimensions: M=1024, N=1024, K=1024
  ...
  Mode: prof
  Device ID: 0
==========================================
[1/4] Performance mode: Skipping data generation
[2/4] Compiling SPMM kernel and host code...
[3/4] Running SPMM kernel...
  Running performance analysis with msprof...
[4/4] Processing profiling results...
SPMM execution completed successfully!

Performance Summary:
  Profiling data: prof/
    - prof/op_summary_20260116_143022.csv
    - prof/ai_core_utilization_20260116_143022.json
```

#### 示例4: 大规模性能测试

```bash
# 4096×4096 × 4096，耗时约10-20秒
./run.sh 0 0 4096 4096 4096 4096 4096 4096 128 prof 0
```

### 误差输出模式

#### 示例5: 生成性能数据CSV

```bash
./run.sh 0 0 2048 2048 2048 2048 2048 2048 128 error 0
```

**特点**:
- `error`: 输出每个元素的误差
- 用于收集性能vs精度的数据
- 生成CSV格式的误差报告

## 性能分析

### 步骤1: 运行性能测试

```bash
# 在后台运行大规模性能测试（推荐）
nohup ./run.sh 0 0 4096 4096 4096 4096 4096 4096 128 prof 0 > perf_4096.log 2>&1 &

# 查看进度
tail -f perf_4096.log
```

### 步骤2: 分析性能数据

```bash
# 查看prof.py处理结果
cat prof/op_summary_*.csv
```

**性能指标**:
```
==========================================
SPMM Performance Summary
==========================================
Matrix Dimensions: M=4096, N=4096, K=4096
Leading Dimensions: lda=4096, ldb=4096, ldc=4096
BSR Block Size: 128
--------------------------------------------------------
Operation Count: 21
Total Time: 15.234 ms
Average Time: 0.725 ms
Min Time: 0.689 ms
Max Time: 0.812 ms
--------------------------------------------------------
Theoretical FLOPs (dense): 137.4 GFLOPs
Actual FLOPs (sparse): 13.74 GFLOPs
Effective Throughput: 1895.6 GFLOPS
==========================================
```

### 步骤3: 对比不同配置

```bash
# 测试不同block size
for bs in 64 128 256; do
    echo "Testing block_size=$bs"
    ./run.sh 0 0 1024 1024 1024 1024 1024 1024 $bs prof 0
done

# 提取吞吐量数据
grep "Effective Throughput" *.log
```

## 错误处理

### 常见错误

#### 错误1: "libascendcl.so: cannot open shared object file"

**症状**:
```
./build/spmm: error while loading shared libraries: libascendcl.so: cannot open shared object file
```

**解决**:
```bash
# 设置库路径
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH

# 永久生效（添加到~/.bashrc）
echo 'export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 错误2: "ccec: command not found"

**症状**:
```
make: ccec: Command not found
```

**解决**:
```bash
# 检查CANN环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 验证
ccec --version
```

#### 错误3: "msprof: command not found"

**症状**:
```
./run.sh: line 46: msprof: command not found
```

**解决**:
```bash
# 添加到PATH
export PATH=/usr/local/Ascend/tools/msprof/bin:$PATH
```

#### 错误4: "No NPU devices found"

**症状**:
```
npu-smi info -l | grep "NPU ID"
# (no output)
```

**解决**:
```bash
# 检查驱动
lsmod | grep drv_dev

# 如果没有输出，加载驱动
sudo modprobe drv_dev

# 检查NPU状态
npu-smi info
```

#### 错误5: Kernel启动失败

**症状**:
```
rtKernelLaunch failed: error code xxx
```

**可能原因**:
1. Kernel名称不匹配（spmm_kernel_aic vs spmm_kernel_aiv）
2. 参数结构不对齐
3. NPU内存不足

**调试步骤**:
```bash
# 1. 检查kernel符号
cd build
strings spmm_kernel.o | grep spmm

# 应该看到：
# spmm_kernel_aic
# spmm_kernel_aiv

# 2. 检查kernel参数大小
echo "Kernel args size: $(sizeof AicKernelArgs)"
echo "Kernel args size: $(sizeof AivKernelArgs)"

# 3. 增加日志级别
export ASCEND_GLOBAL_LOG_LEVEL=0
./run.sh 0 0 64 64 64 64 64 64 64
```

## 高级用法

### 1. 手动执行各步骤

```bash
cd ascblas/src

# 1. 生成数据
python3 spmm_gen_data.py 0 0 1024 1024 1024 1024 1024 1024 128

# 2. 编译
make clean && make

# 3. 运行 (生成数据后，data/目录包含.bin文件)
cd build
./spmm 0 0 1024 1024 1024 1024 1024 1024 128 1 0

# 4. 性能分析（单独使用msprof）
cd ..
msprof --application="./build/spmm 0 0 1024 1024 1024 1024 1024 1024 128 0 0" \
           --output=prof --ai-core=on
```

### 2. 多卡测试

```bash
# 在卡0上运行
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128 prof 0

# 在卡1上运行
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128 prof 1

# 并行运行（两个终端）
# Terminal 1:
export ASCEND_DEVICE_ID=0
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128 prof 0

# Terminal 2:
export ASCEND_DEVICE_ID=1
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128 prof 1
```

### 3. 性能优化

```bash
# 设置高性能模式
sudo npu-smi set -t high-performance -i 0

# 监控频率
watch -n 1 npu-smi info -m cpu -i 0

# 设置独占模式（避免其他进程干扰）
export ASCEND_VISIBLE_DEVICES=0
./run.sh 0 0 4096 4096 4096 4096 4096 4096 128 prof 0
```

### 4. 批量测试

创建 `batch_test.sh`:
```bash
#!/bin/bash
# 批量性能测试

sizes="512 1024 2048 4096"
block_size=128

echo "M,N,K,Time_ms,GFLOPS,Sparsity" > results.csv

for size in $sizes; do
    echo "Testing size: $size"
    ./run.sh 0 0 $size $size $size $size $size $size $block_size prof 0 > tmp.log

    time_ms=$(grep "Average Time:" tmp.log | awk '{print $3}')
    gflops=$(grep "Effective Throughput:" tmp.log | awk '{print $3}')

    echo "$size,$size,$size,$time_ms,$gflops,90%" >> results.csv
done

echo "Results saved to results.csv"
```

## 清理

```bash
# 清理所有中间文件
cd ascblas/src
make clean
rm -rf data/
rm -rf prof/
rm -rf build/

# 或者使用脚本
./clean_all.sh
```

## 测试矩阵

### 推荐的测试配置

| 测试用例 | M | N | K | Block Size | 预期稀疏度 | 预期性能 |
|---------|---|---|---|------------|----------|---------|
| 小型测试 | 64 | 64 | 64 | 64 | 85% | ~50 GFLOPS |
| 功能验证 | 128 | 128 | 128 | 128 | 90% | ~200 GFLOPS |
| 性能基准 | 1024 | 1024 | 1024 | 128 | 90% | ~1800 GFLOPS |
| 大规模测试 | 4096 | 4096 | 4096 | 128 | 90% | ~2800 GFLOPS |
| 矩形矩阵 | 2048 | 1024 | 4096 | 128 | 90% | ~1500 GFLOPS |

### 完整测试流程

```bash
cd ascblas/src

# 1. 环境检查
./check_env.sh

# 2. 快速构建
./build.sh

# 3. 功能验证（从小到大）
./run.sh 0 0 128 128 128 128 128 128 128
./run.sh 0 0 512 512 512 512 512 512 128
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128

# 4. 性能测试
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128 prof 0
./run.sh 0 0 2048 2048 2048 2048 2048 2048 128 prof 0

# 5. 大规模测试
./run.sh 0 0 4096 4096 4096 4096 4096 4096 128 prof 0
```

## 总结

### 关键命令备忘录

```bash
# 编译
make clean && make

# 功能测试
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128

# 性能测试
./run.sh 0 0 1024 1024 1024 1024 1024 1024 128 prof 0

# 环境检查
./check_env.sh

# 快速构建与测试
./build.sh
```

### 成功标志

1. **编译成功**: `build/spmm` 和 `build/spmm_kernel.o` 生成
2. **功能验证**: "All data is correct!" 或 "BSR SPMM Verification passed"
3. **性能**: 1024×1024矩阵达到1000+ GFLOPS
4. **稀疏度**: 自动检测约90%稀疏度

### 下一步

- 参考 `README.md` 了解更多优化技巧
- 查看 `tests/ascblasHgemm/README.md` 了解原始GEMM文档
- 调整 `spmm_gen_data.py` 中的稀疏度参数
- 尝试不同的block size（64/128/256）

---

**最后更新**: 2026-01-16
**维护**: Ascend SPMM Development Team
