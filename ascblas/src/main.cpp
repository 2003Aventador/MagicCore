#include <acl/acl.h>
#include "spmm.h"
#include "data_utils.h"
#include "sr_bcrs_utils.h"
#include <vector>
#include <iostream> // 为了使用 std::cerr 等

/*
verifyLevel 表示测试的等级：
0 表示测试算子性能
1 表示测试单一输入的正确性
2 表示测试输入的正确性，并输出误差供csv收集
*/
int verifyLevel = 0;

// deviceId 表示程序运行在第几号卡上
int deviceId = 0;

int main(int argc, char** argv)
{
    // 参数检查
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <batch> <M> <N> <K>" << std::endl;
        return -1;
    }
    
    // 获取参数
    int32_t M = std::stoi(argv[1]);
    int32_t N = std::stoi(argv[2]);
    int32_t K = std::stoi(argv[3]);

    const int vec_length = 16;
    const int K0 = 128;
    const int N0 = 128;
    // stride 表示一个tile_A块中的向量个数，用于sparse_to_sr_bcrs转换
    const int stride = K0;

    // 原始矩阵大小
    size_t A_size = M * K * sizeof(__fp16);
    size_t B_size = K * N * sizeof(__fp16);
    size_t C_size = M * N * sizeof(__fp16);

    CALL_RT(aclInit(nullptr));
    CALL_RT(aclrtSetDevice(deviceId));

    // 调用ascblas算子
    aclrtStream stream;
    ascblasHandle_t handle;
    ascblasCreate(&handle);
    ascblasGetStream(handle, &stream);

    __fp16* h_A = nullptr; 
    __fp16* h_B = nullptr;
    __fp16* h_C = nullptr;
    // __fp16* h_C_ref = nullptr;  // 参考结果（用于验证），暂未使用

    std::vector<__fp16> values;
    std::vector<int32_t> row_indices;
    std::vector<int32_t> col_indices;
    std::vector<int32_t> row_offsets;

    __fp16* d_values = nullptr;
    int32_t* d_row_indices = nullptr;
    int32_t* d_col_indices = nullptr;
    int32_t* d_row_offsets = nullptr;
    __fp16* d_B = nullptr;
    __fp16* d_C = nullptr;


    CALL_RT(aclrtMallocHost((void**)(&h_A), A_size));
    CALL_RT(aclrtMallocHost((void**)(&h_B), B_size));
    CALL_RT(aclrtMallocHost((void**)(&h_C), C_size));
    // 读取数据
    ReadFile("../data/A.bin", h_A, A_size);
    ReadFile("../data/B.bin", h_B, B_size);
    ReadFile("../data/C.bin", h_C, C_size);

    sparse_to_sr_bcrs(vec_length, stride, row_indices, row_offsets, col_indices, values, h_A);

    // 计算size
    // values: nnz_vectors * vec_length * sizeof(half)
    int nnz_vectors = values.size() / vec_length;
    int values_size = values.size() * sizeof(__fp16);
    int row_indices_size = row_indices.size() * sizeof(int32_t);
    int col_indices_size = col_indices.size() * sizeof(int32_t);
    int row_offsets_size = row_offsets.size() * sizeof(int32_t);

    // 申请Device内存
    CALL_RT(aclrtMalloc((void**)(&d_values), values_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void**)(&d_row_indices), row_indices_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void**)(&d_col_indices), col_indices_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void**)(&d_row_offsets), row_offsets_size, ACL_MEM_MALLOC_HUGE_FIRST));

    // 将数据从Host拷贝到Device
    CALL_RT(aclrtMemcpy(d_values, values_size, values.data(), values_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_row_indices, row_indices_size, row_indices.data(), row_indices_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_col_indices, col_indices_size, col_indices.data(), col_indices_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_row_offsets, row_offsets_size, row_offsets.data(), row_offsets_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_B, B_size, h_B, B_size, ACL_MEMCPY_HOST_TO_DEVICE));

    // 申请d_C的Device内存
    CALL_RT(aclrtMalloc((void**)(&d_C), C_size, ACL_MEM_MALLOC_HUGE_FIRST));

    CALL_RT(ascblasSpmm(
        handle,
        M,
        N,
        K,
        vec_length,
        K0,
        N0,
        d_values,
        d_row_indices,
        d_col_indices,
        d_row_offsets,
        d_B,
        d_C
    ));


    CALL_RT(aclrtSynchronizeStream(stream));

    CALL_RT(aclrtFree(d_values));
    CALL_RT(aclrtFree(d_row_indices));
    CALL_RT(aclrtFree(d_col_indices));
    CALL_RT(aclrtFree(d_row_offsets));
    CALL_RT(aclrtFree(d_B));
    CALL_RT(aclrtFree(d_C));
    CALL_RT(aclrtFreeHost(h_A));
    CALL_RT(aclrtFreeHost(h_B));
    CALL_RT(aclrtFreeHost(h_C));
    // CALL_RT(aclrtFreeHost(h_C_ref));
    CALL_RT(aclrtResetDevice(deviceId));
    CALL_RT(aclFinalize());

    return 0;
}
