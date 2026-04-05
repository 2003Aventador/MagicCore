#include <acl/acl.h>
#include "spmm.h"
#include "sr_bcrs_utils.h"
#include "file_utils.h"
#include "data_utils.h"
#include <vector>
#include <iostream>

int deviceId = 0;
int verifyLevel = 0;

namespace {
constexpr int32_t kCubeKAlign = 16;

void BuildAlignedSrBcrs(
    int32_t vec_length,
    const std::vector<__fp16> &src_values,
    const std::vector<int32_t> &src_row_indices,
    const std::vector<int32_t> &src_col_indices,
    std::vector<__fp16> &aligned_values,
    std::vector<int32_t> &aligned_row_indices,
    std::vector<int32_t> &aligned_col_indices,
    std::vector<int32_t> &row_offsets)
{
    const int32_t num_vec_rows = static_cast<int32_t>(src_row_indices.size()) - 1;
    aligned_row_indices.resize(num_vec_rows + 1, 0);
    row_offsets.resize(num_vec_rows + 1, 0);

    int32_t total_aligned_vectors = 0;
    for (int32_t vec_row = 0; vec_row < num_vec_rows; ++vec_row) {
        const int32_t row_start = src_row_indices[vec_row];
        const int32_t row_end = src_row_indices[vec_row + 1];
        const int32_t nnz_vectors = row_end - row_start;
        const int32_t padded_vectors =
            ((nnz_vectors + kCubeKAlign - 1) / kCubeKAlign) * kCubeKAlign;
        total_aligned_vectors += padded_vectors;
    }

    aligned_values.reserve(static_cast<size_t>(total_aligned_vectors) * vec_length);
    aligned_col_indices.reserve(total_aligned_vectors);

    for (int32_t vec_row = 0; vec_row < num_vec_rows; ++vec_row) {
        const int32_t row_start = src_row_indices[vec_row];
        const int32_t row_end = src_row_indices[vec_row + 1];
        const int32_t nnz_vectors = row_end - row_start;
        const int32_t padded_vectors =
            ((nnz_vectors + kCubeKAlign - 1) / kCubeKAlign) * kCubeKAlign;

        row_offsets[vec_row] = vec_row * vec_length;
        aligned_row_indices[vec_row] = static_cast<int32_t>(aligned_col_indices.size());

        for (int32_t elem = row_start; elem < row_end; ++elem) {
            aligned_col_indices.push_back(src_col_indices[elem]);
            const int32_t value_offset = elem * vec_length;
            aligned_values.insert(
                aligned_values.end(),
                src_values.begin() + value_offset,
                src_values.begin() + value_offset + vec_length);
        }

        for (int32_t pad = nnz_vectors; pad < padded_vectors; ++pad) {
            aligned_col_indices.push_back(0);
            aligned_values.insert(aligned_values.end(), vec_length, static_cast<__fp16>(0.0f));
        }

        aligned_row_indices[vec_row + 1] = static_cast<int32_t>(aligned_col_indices.size());
    }

    row_offsets[num_vec_rows] = num_vec_rows * vec_length;
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> [verifyLevel] [deviceId]" << std::endl;
        return -1;
    }

    int32_t M = std::stoi(argv[1]);
    int32_t N = std::stoi(argv[2]);
    int32_t K = std::stoi(argv[3]);
    if (argc > 4) {
        verifyLevel = std::stoi(argv[4]);
    }
    if (argc > 5) {
        deviceId = std::stoi(argv[5]);
    }

    const int vec_length = 16;
    const int K0 = 128;
    const int N0 = 128;

    if (N % kCubeKAlign != 0) {
        std::cerr << "N must be a multiple of 16 for the current cube path." << std::endl;
        return -1;
    }

    const size_t B_count = static_cast<size_t>(K) * N;
    const size_t C_count = static_cast<size_t>(M) * N;
    const size_t B_size = B_count * sizeof(__fp16);
    const size_t C_size = C_count * sizeof(float);

    CALL_RT(aclInit(nullptr));
    CALL_RT(aclrtSetDevice(deviceId));

    aclrtStream stream;
    ascblasHandle_t handle;
    ascblasCreate(&handle);
    ascblasGetStream(handle, &stream);

    __fp16 *h_B = nullptr;
    float *h_C = nullptr;
    float *h_C_ref = nullptr;
    CALL_RT(aclrtMallocHost(reinterpret_cast<void **>(&h_B), B_size));
    CALL_RT(aclrtMallocHost(reinterpret_cast<void **>(&h_C), C_size));
    CALL_RT(aclrtMallocHost(reinterpret_cast<void **>(&h_C_ref), C_size));

    std::vector<__fp16> values;
    std::vector<int32_t> row_indices;
    std::vector<int32_t> col_indices;
    std::vector<int32_t> row_offsets;

    int32_t A_num_vectors = 0;
    int32_t A_d = 0;
    ReadAMatrixVectorCSR(M, A_d, values, col_indices, row_indices, A_num_vectors);
    ReadFloat32ToFp16("../../data/B_dense.bin", h_B, B_count);
    ReadBinFile<float>("../../data/C_dense.bin", h_C_ref, C_count);

    if (A_d != vec_length) {
        std::cerr << "Unsupported vector length in SR-BCRS input: " << A_d << std::endl;
        return -1;
    }

    std::vector<__fp16> aligned_values;
    std::vector<int32_t> aligned_row_indices;
    std::vector<int32_t> aligned_col_indices;
    BuildAlignedSrBcrs(
        vec_length,
        values,
        row_indices,
        col_indices,
        aligned_values,
        aligned_row_indices,
        aligned_col_indices,
        row_offsets);

    values.swap(aligned_values);
    row_indices.swap(aligned_row_indices);
    col_indices.swap(aligned_col_indices);

    const size_t values_size = values.size() * sizeof(__fp16);
    const size_t row_indices_size = row_indices.size() * sizeof(int32_t);
    const size_t col_indices_size = col_indices.size() * sizeof(int32_t);
    const size_t row_offsets_size = row_offsets.size() * sizeof(int32_t);

    __fp16 *d_values = nullptr;
    int32_t *d_row_indices = nullptr;
    int32_t *d_col_indices = nullptr;
    int32_t *d_row_offsets = nullptr;
    __fp16 *d_B = nullptr;
    float *d_C = nullptr;

    CALL_RT(aclrtMalloc(reinterpret_cast<void **>(&d_values), values_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc(reinterpret_cast<void **>(&d_row_indices), row_indices_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc(reinterpret_cast<void **>(&d_col_indices), col_indices_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc(reinterpret_cast<void **>(&d_row_offsets), row_offsets_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc(reinterpret_cast<void **>(&d_B), B_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc(reinterpret_cast<void **>(&d_C), C_size, ACL_MEM_MALLOC_HUGE_FIRST));

    CALL_RT(aclrtMemcpy(d_values, values_size, values.data(), values_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_row_indices, row_indices_size, row_indices.data(), row_indices_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_col_indices, col_indices_size, col_indices.data(), col_indices_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_row_offsets, row_offsets_size, row_offsets.data(), row_offsets_size, ACL_MEMCPY_HOST_TO_DEVICE));
    CALL_RT(aclrtMemcpy(d_B, B_size, h_B, B_size, ACL_MEMCPY_HOST_TO_DEVICE));

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

    CALL_RT(aclrtSynchronizeStream(stream));

    CALL_RT(aclrtMemcpy(h_C, C_size, d_C, C_size, ACL_MEMCPY_DEVICE_TO_HOST));

    const bool verify_passed = CompareFloat32Buffers(h_C, h_C_ref, C_count);
    std::cout << "Verification result: "
              << (verify_passed ? "PASS" : "FAIL")
              << std::endl;

    CALL_RT(aclrtFree(d_values));
    CALL_RT(aclrtFree(d_row_indices));
    CALL_RT(aclrtFree(d_col_indices));
    CALL_RT(aclrtFree(d_row_offsets));
    CALL_RT(aclrtFree(d_B));
    CALL_RT(aclrtFree(d_C));
    CALL_RT(aclrtFreeHost(h_B));
    CALL_RT(aclrtFreeHost(h_C));
    CALL_RT(aclrtFreeHost(h_C_ref));
    CALL_RT(aclrtResetDevice(deviceId));
    CALL_RT(aclFinalize());
    return verify_passed ? 0 : 1;
}
