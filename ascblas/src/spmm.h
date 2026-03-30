#include <fstream>
#include "ascblas.h"

aclError ascblasSpmm(
    ascblasHandle_t handle,
    int M,
    int N,
    int K,
    int vec_length,
    int K0,
    int N0,
    __fp16* values,
    int* row_indices,
    int* col_indices,
    int* row_offsets,
    __fp16 *B,
    float *C
)
{
    aclError error;
    aclrtStream stream;

    // 得到stream
    ascblasGetStream(handle, &stream);
    
    std::string kernel_name = "spmm_kernel";
    std::string bin_name = "spmm.o";
    RegisterBinaryKernel(kernel_name.c_str(), bin_name.c_str());

    int64_t num_row_blocks = (M + vec_length - 1) / vec_length;  // A的行块数
    // 为每行分配一个核处理该行的所有非零块
    int64_t groupDim = num_row_blocks < CORENUM ? num_row_blocks : CORENUM;

    // 合并后的Kernel参数结构体（参考ascblasHgemm）
    typedef struct {
        int M;
        int N;
        int K;
        int vec_length;
        int K0;
        int N0;
        __fp16 *values;
        int *row_indices;
        int *col_indices;
        int *row_offsets;
        __fp16 *B;
        float *C;
    } KernelArgs;

    // 准备Kernel参数（合并为一个）
    KernelArgs kernel_args;
    kernel_args.M = M;
    kernel_args.N = N;
    kernel_args.K = K;
    kernel_args.vec_length=vec_length;
    kernel_args.K0=K0;
    kernel_args.N0=N0;
    kernel_args.values = values;
    kernel_args.row_indices = row_indices;
    kernel_args.col_indices = col_indices;
    kernel_args.row_offsets = row_offsets;
    kernel_args.B = B;
    kernel_args.C = C;

    error = rtKernelLaunch((void *)kernel_name.c_str(), groupDim, &kernel_args, sizeof(kernel_args), NULL, stream);

    aclrtSynchronizeStream(stream);
    return error;
}
