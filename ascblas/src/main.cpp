#include <acl/acl.h>
#include "spmm.h"
#include "data_utils.h"
#include "sr_bcrs_utils.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdint>

/*
verifyLevel 表示测试的等级：
0 表示测试算子性能
1 表示测试单一输入的正确性
2 表示测试输入的正确性，并输出误差供csv收集
*/
int verifyLevel = 0;

// deviceId 表示程序运行在第几号卡上
int deviceId = 0;

// ===================== 精简版工具函数 =====================
template <typename T>
void ReadBinFile(const std::string &file_path, T *data, size_t count)
{
    /**
     * @brief 通用二进制文件读取函数（支持任意类型）
     * @tparam T 数据类型
     * @param file_path 文件路径
     * @param data 输出数据指针
     * @param count 要读取的元素个数
     */
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: 无法打开二进制文件 " << file_path << std::endl;
        exit(1);
    }

    file.read(reinterpret_cast<char *>(data), count * sizeof(T));
    if (!file)
    {
        std::cerr << "Error: 读取文件 " << file_path << " 失败，读取字节数: " << file.gcount()
                  << " 期望字节数: " << count * sizeof(T) << std::endl;
        file.close();
        exit(1);
    }

    file.close();
    std::cout << "✅ 成功读取二进制文件: " << file_path << " (元素数: " << count << ")" << std::endl;
}


void ReadFloat32ToFp16(const std::string &file_path, __fp16 *data_fp16, size_t count)
{
/**
 * @brief 读取float32格式的矩阵并转换为__fp16
 * @param file_path 二进制文件路径
 * @param data_fp16 输出：__fp16格式矩阵
 * @param count 元素个数
 */
    // 1. 读取float32数据
    float *data_float = new float[count];
    ReadBinFile<float>(file_path, data_float, count);

    // 2. 转换为__fp16
    for (size_t i = 0; i < count; i++)
    {
        data_fp16[i] = (__fp16)data_float[i];
    }

    // 3. 打印前16个元素验证
    std::cout << ">>> 前16个元素（__fp16）:" << std::endl;
    for (size_t i = 0; i < 16 && i < count; i++)
    {
        printf("%.4f ", (float)data_fp16[i]);
        if ((i + 1) % 8 == 0)
            printf("\n");
    }
    printf("\n");

    delete[] data_float;
}


int32_t ReadMetaValue(const std::string &file_path, const std::string &key)
{
/**
 * @brief 读取文本元信息文件（仅保留核心功能）
 * @param file_path 元信息文件路径
 * @param key 要读取的键名
 * @return 对应的值
 */
    std::ifstream meta_file(file_path);
    if (!meta_file.is_open())
    {
        std::cerr << "Error: 无法打开元信息文件 " << file_path << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(meta_file, line))
    {
        size_t colon_pos = line.find(":");
        if (colon_pos == std::string::npos)
            continue;

        std::string line_key = line.substr(0, colon_pos);
        line_key.erase(0, line_key.find_first_not_of(" \t"));
        line_key.erase(line_key.find_last_not_of(" \t") + 1);

        if (line_key == key)
        {
            std::string value_str = line.substr(colon_pos + 1);
            value_str.erase(0, value_str.find_first_not_of(" \t"));
            meta_file.close();
            return std::stoi(value_str);
        }
    }

    meta_file.close();
    std::cerr << "Error: 在元信息文件中未找到键 " << key << std::endl;
    exit(1);
}


void ReadAMatrixVectorCSR(int32_t M, int32_t &d,
                          std::vector<__fp16> &vec_csr_data,
                          std::vector<int32_t> &vec_csr_cols,
                          std::vector<int32_t> &vec_csr_indptr,
                          int32_t &num_vectors)
{
/**
 * @brief 读取A矩阵的vector-CSR格式（保留核心功能）
 * @param M 矩阵行数
 * @param d 向量块维度
 * @param vec_csr_data 输出：vector-CSR数值（__fp16）
 * @param vec_csr_cols 输出：vector-CSR列索引
 * @param vec_csr_indptr 输出：vector-CSR行指针
 * @param num_vectors 输出：向量块数量
 */
    // 1. 读取vector-CSR元信息
    num_vectors = ReadMetaValue("../data/vector_csr/A_vector_csr_meta.txt", "num_vector_blocks");
    d = ReadMetaValue("../data/vector_csr/A_vector_csr_meta.txt", "d");
    int32_t row_blocks = ReadMetaValue("../data/vector_csr/A_vector_csr_meta.txt", "row_blocks_count");

    // 2. 读取并转换vector-CSR data
    size_t vec_data_count = num_vectors * d;
    float *vec_data_float = new float[vec_data_count];
    ReadBinFile<float>("../data/vector_csr/A_data.bin", vec_data_float, vec_data_count);

    vec_csr_data.resize(vec_data_count);
    for (size_t i = 0; i < vec_data_count; i++)
    {
        vec_csr_data[i] = (__fp16)vec_data_float[i];
    }

    // 3. 读取vector-CSR cols和indptr
    vec_csr_cols.resize(num_vectors);
    ReadBinFile<int32_t>("../data/vector_csr/A_cols.bin", vec_csr_cols.data(), num_vectors);

    vec_csr_indptr.resize(row_blocks + 1);
    ReadBinFile<int32_t>("../data/vector_csr/A_indptr.bin", vec_csr_indptr.data(), row_blocks + 1);

    // 4. 打印验证信息
    std::cout << "\n>>> A矩阵vector-CSR格式信息:" << std::endl;
    std::cout << "向量块数量: " << num_vectors << "  向量维度d: " << d << std::endl;
    std::cout << "行块数: " << row_blocks << "  行指针长度: " << vec_csr_indptr.size() << std::endl;

    delete[] vec_data_float;
}
// ==================================================================

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
    const int stride = K0;

    // 矩阵大小（__fp16格式）
    size_t A_count = M * K;
    size_t B_count = K * N;
    size_t C_count = M * N;
    size_t A_size = A_count * sizeof(__fp16);
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

    // 主机端内存分配
    __fp16 *h_A = nullptr;
    __fp16 *h_B = nullptr;
    __fp16 *h_C = nullptr;
    CALL_RT(aclrtMallocHost((void **)(&h_A), A_size));
    CALL_RT(aclrtMallocHost((void **)(&h_B), B_size));
    CALL_RT(aclrtMallocHost((void **)(&h_C), C_size));

    // ===================== 核心读取逻辑（精简版） =====================
    // 1. 读取A矩阵（float32 → __fp16）
    // std::cout << "\n=== 读取A矩阵（密集格式）===" << std::endl;
    // ReadFloat32ToFp16("../data/A_dense.bin", h_A, A_count);

    // 【可选】读取A矩阵vector-CSR格式（按需启用）
    std::vector<__fp16> A_vec_csr_data;
    std::vector<int32_t> A_vec_csr_cols, A_vec_csr_indptr;
    int32_t A_num_vectors = 0, A_d = 0; // 调用 ReadAMatrixVectorCSR 后，A_num_vectors 会被赋值
    ReadAMatrixVectorCSR(M, A_d, A_vec_csr_data, A_vec_csr_cols, A_vec_csr_indptr, A_num_vectors);

    // 2. 读取B矩阵（float32 → __fp16）
    std::cout << "\n=== 读取B矩阵（密集格式）===" << std::endl;
    ReadFloat32ToFp16("../data/B_dense.bin", h_B, B_count);

    // 3. 读取参考结果C
    std::cout << "\n=== 读取参考结果C矩阵 ===" << std::endl;
    // ReadBinFile<__fp16>("../data/C_dense.bin", h_C, C_count);
    ReadFloat32ToFp16("../data/C_dense.bin", h_C, C_count);
    // ==================================================================

    // SR-BCRS格式转换（保留原有逻辑）
    std::vector<__fp16> values;
    std::vector<int32_t> row_indices;
    std::vector<int32_t> col_indices;
    std::vector<int32_t> row_offsets;
    std::cout << "\n=== 转换A矩阵为SR-BCRS格式 ===" << std::endl;
    sparse_to_sr_bcrs(vec_length, stride, row_indices, row_offsets, col_indices, values, h_A);

    // 计算设备端内存大小
    int nnz_vectors = values.size() / vec_length;
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
    __fp16 *d_C = nullptr;

    CALL_RT(aclrtMalloc((void **)(&d_values), values_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&d_row_indices), row_indices_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&d_col_indices), col_indices_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&d_row_offsets), row_offsets_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&d_B), B_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&d_C), C_size, ACL_MEM_MALLOC_HUGE_FIRST));

    // 主机→设备数据拷贝
    std::cout << "\n=== 拷贝数据到设备端 ===" << std::endl;
    CALL_RT(aclrtMemcpy(d_values, values_size, values.data(), values_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_row_indices, row_indices_size, row_indices.data(), row_indices_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_col_indices, col_indices_size, col_indices.data(), col_indices_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_row_offsets, row_offsets_size, row_offsets.data(), row_offsets_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_B, B_size, h_B, B_size, ACL_MEMCPY_HOST_TO_DEVICE));

    // 调用SPMM算子
    std::cout << "\n=== 调用ascblasSpmm算子 ===" << std::endl;
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
    std::cout << "\n=== 释放内存资源 ===" << std::endl;
    CALL_RT(aclrtFree(d_values));
    CALL_RT(aclrtFree(d_row_indices));
    CALL_RT(aclrtFree(d_col_indices));
    CALL_RT(aclrtFree(d_row_offsets));
    CALL_RT(aclrtFree(d_B));
    CALL_RT(aclrtFree(d_C));
    CALL_RT(aclrtFreeHost(h_A));
    CALL_RT(aclrtFreeHost(h_B));
    CALL_RT(aclrtFreeHost(h_C));

    // 清理ACL资源
    CALL_RT(aclrtResetDevice(deviceId));
    CALL_RT(aclFinalize());

    std::cout << "\n✅ 程序执行完成！" << std::endl;
    return 0;
}