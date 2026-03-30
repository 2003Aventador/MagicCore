#include <acl/acl.h>
#include "spmm.h"
#include "sr_bcrs_utils.h"
#include "file_utils.h"
#include "data_utils.h"
#include <vector>
#include <iostream>

// deviceId 表示程序运行在第几号卡上
int deviceId = 0;

int verifyLevel = 0; // 0: 不验证结果，1: 验证结果并打印错误信息，2: 验证结果并打印详细对比信息

int main(int argc, char **argv)
{
    // 参数检查
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> [verifyLevel] [deviceId]" << std::endl;
        return -1;
    }

    // 获取参数
    int32_t M = std::stoi(argv[1]);
    int32_t N = std::stoi(argv[2]);
    int32_t K = std::stoi(argv[3]);
    if (argc > 4)
        verifyLevel = std::stoi(argv[4]);
    if (argc > 5)
        deviceId = std::stoi(argv[5]);

    const int vec_length = 16;
    const int K0 = 128;
    const int N0 = 128;

    // 矩阵大小（__fp16格式）
    size_t B_count = K * N;
    size_t C_count = M * N;
    size_t B_size = B_count * sizeof(__fp16);
    size_t C_size = C_count * sizeof(__fp16);

    // 初始化ACL
    CALL_RT(aclInit(nullptr));
    CALL_RT(aclrtSetDevice(deviceId));

    // 创建算子句柄和流
    aclrtStream stream;
    ascblasHandle_t handle;
    ascblasCreate(&handle);
    ascblasGetStream(handle, &stream);

    __fp16 *h_B = nullptr;
    float *h_C = nullptr;
    CALL_RT(aclrtMallocHost((void **)(&h_B), B_size));
    CALL_RT(aclrtMallocHost((void **)(&h_C), C_size));

    std::vector<__fp16> values;
    std::vector<int32_t> row_indices;
    std::vector<int32_t> col_indices;
    std::vector<int32_t> row_offsets;
    
    int32_t A_num_vectors = 0, A_d = 0;
    ReadAMatrixVectorCSR(M, A_d, values, col_indices, row_indices, A_num_vectors);
    ReadFloat32ToFp16("../data/B_dense.bin", h_B, B_count);
    ReadFloat32ToFp16("../data/C_dense.bin", h_C, C_count);

    // row_indices与row_offsets相同（简化处理）
    row_offsets = row_indices;

    // 计算设备端内存大小
    int values_size = values.size() * sizeof(__fp16);
    int row_indices_size = row_indices.size() * sizeof(int32_t);
    int col_indices_size = col_indices.size() * sizeof(int32_t);
    int row_offsets_size = row_offsets.size() * sizeof(int32_t);

    // 设备端内存分配
    __fp16 *d_values = nullptr;
    int32_t *d_row_indices = nullptr;
    int32_t *d_col_indices = nullptr;
    int32_t *d_row_offsets = nullptr;
    __fp16 *d_B = nullptr;
    float *d_C = nullptr;

    CALL_RT(aclrtMalloc((void **)(&d_values), values_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&d_row_indices), row_indices_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&d_col_indices), col_indices_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&d_row_offsets), row_offsets_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&d_B), B_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&d_C), C_size, ACL_MEM_MALLOC_HUGE_FIRST));

    // 主机→设备数据拷贝
    CALL_RT(aclrtMemcpy(d_values, values_size, values.data(), values_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_row_indices, row_indices_size, row_indices.data(), row_indices_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_col_indices, col_indices_size, col_indices.data(), col_indices_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_row_offsets, row_offsets_size, row_offsets.data(), row_offsets_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_B, B_size, h_B, B_size, ACL_MEMCPY_HOST_TO_DEVICE));

    // 调用SPMM算子
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
        d_C));

    // 同步流等待算子执行完成
    CALL_RT(aclrtSynchronizeStream(stream));

    // 内存释放
    CALL_RT(aclrtFree(d_values));
    CALL_RT(aclrtFree(d_row_indices));
    CALL_RT(aclrtFree(d_col_indices));
    CALL_RT(aclrtFree(d_row_offsets));
    CALL_RT(aclrtFree(d_B));
    CALL_RT(aclrtFree(d_C));
    CALL_RT(aclrtFreeHost(h_B));
    CALL_RT(aclrtFreeHost(h_C));
    CALL_RT(aclrtResetDevice(deviceId));
    CALL_RT(aclFinalize());
    return 0;
}