#include "kernel_operator.h"
#include "stdio.h"

#define NUM_ELE_PERBLOCK 16
#define CUBE_M0 16
#define CUBE_N0 16
#define CUBE_K0 16
#define CUBE_MATRIX_SIZE (16 * 16)
#define DIV_UP(v, n) (((v) + (n) - 1) / (n))

__aicore__ __inline__ void ascblasSpmmAIC(
    int M,
    int N,
    int K,
    int vec_length,
    int K0,
    int N0,
    __gm__ half * __restrict__ values,
    __gm__ int * __restrict__ row_indices,
    __gm__ int * __restrict__ col_indices,
    __gm__ int * __restrict__ row_offsets,
    __gm__ half * __restrict__ B,
    __gm__ float * __restrict__ C)
{
    (void)K;
    auto L1_base_a = reinterpret_cast<__cbuf__ __fp16 *>((uintptr_t)0);
    auto L1_base_b = reinterpret_cast<__cbuf__ __fp16 *>((uintptr_t)(128 * 1024));
    auto L0A_base = reinterpret_cast<__ca__ __fp16 *>((uintptr_t)0);
    auto L0B_base = reinterpret_cast<__cb__ __fp16 *>((uintptr_t)0);
    auto L0C_base = reinterpret_cast<__cc__ float *>((uintptr_t)0);

    const int m_vec = M / vec_length;
    const uint64_t nz2nd_config = 4295032833ULL;
    bool pending_store = false;

    for (int64_t vec_row = 0; vec_row < m_vec; ++vec_row) {
        if (vec_row % get_block_num() != get_block_idx()) {
            continue;
        }

        const int row_start = row_indices[vec_row];
        const int row_end = row_indices[vec_row + 1];
        const int nnz_vectors = row_end - row_start;
        if (nnz_vectors == 0) {
            continue;
        }

        const int k_steps = DIV_UP(nnz_vectors, K0);
        const int n_steps = DIV_UP(N, N0);
        const int logical_row = row_offsets[vec_row];
        int index[128];

        for (int n_step = 0; n_step < n_steps; ++n_step) {
            if (pending_store) {
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
                pending_store = false;
            }

            const int remain = min(N0, N - n_step * N0);

            for (int k_step = 0; k_step < k_steps; ++k_step) {
                const int tile_start = row_start + k_step * K0;
                const int cnt = min(row_end - tile_start, K0);
                const int a_repeat = cnt / CUBE_K0;
                const int b_repeat = (cnt * remain) / CUBE_MATRIX_SIZE;

                for (int i = 0; i < cnt; ++i) {
                    index[i] = col_indices[tile_start + i];
                }

                auto dst_A = L1_base_a;
                auto src_A = values + tile_start * vec_length;
                copy_gm_to_cbuf(
                    dst_A,
                    src_A,
                    static_cast<uint8_t>(0),
                    static_cast<uint16_t>(cnt),
                    static_cast<uint16_t>(1),
                    static_cast<uint16_t>(0),
                    static_cast<uint16_t>(0),
                    PAD_NONE);

                auto dst_B = L1_base_b;
                for (int i = 0; i < cnt; ++i) {
                    auto src_B = B + index[i] * N + n_step * N0;
                    auto dst_B_row = dst_B + i * remain;
                    copy_gm_to_cbuf(
                        dst_B_row,
                        src_B,
                        static_cast<uint8_t>(0),
                        static_cast<uint16_t>(1),
                        static_cast<uint16_t>(remain / NUM_ELE_PERBLOCK),
                        static_cast<uint16_t>(0),
                        static_cast<uint16_t>(0),
                        PAD_NONE);
                }

                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

                load_cbuf_to_ca(
                    L0A_base,
                    dst_A,
                    static_cast<uint16_t>(0),
                    static_cast<uint8_t>(a_repeat),
                    static_cast<uint16_t>(1),
                    static_cast<uint16_t>(0),
                    static_cast<uint8_t>(0),
                    true,
                    inc);
                load_cbuf_to_cb(
                    L0B_base,
                    dst_B,
                    static_cast<uint16_t>(0),
                    static_cast<uint8_t>(b_repeat),
                    static_cast<uint16_t>(1),
                    static_cast<uint16_t>(0),
                    static_cast<uint8_t>(0),
                    true,
                    inc);

                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);

                mad(
                    L0C_base,
                    L0A_base,
                    L0B_base,
                    static_cast<uint16_t>(vec_length),
                    static_cast<uint16_t>(cnt),
                    static_cast<uint16_t>(remain),
                    static_cast<uint8_t>(0),
                    static_cast<uint8_t>(0),
                    static_cast<uint8_t>(0),
                    static_cast<bool>(0),
                    static_cast<bool>(0),
                    static_cast<bool>(0),
                    static_cast<bool>(k_step == 0));
            }

            pipe_barrier(PIPE_M);
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID3);

            set_nd_para(nz2nd_config);
            copy_matrix_cc_to_gm(
                C + logical_row * N + n_step * N0,
                L0C_base,
                static_cast<uint8_t>(0),
                static_cast<uint16_t>(remain),
                static_cast<uint16_t>(vec_length),
                static_cast<uint32_t>(N),
                static_cast<uint16_t>(vec_length),
                static_cast<uint8_t>(0),
                static_cast<uint8_t>(0b00000),
                static_cast<uint8_t>(0b000),
                static_cast<bool>(0),
                static_cast<bool>(1));
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
            pending_store = true;
        }
    }

    if (pending_store) {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
    }
    pipe_barrier(PIPE_ALL);
}

#if __DAV_C220_CUBE__
extern "C" __global__ __aicore__ void spmm_kernel_mix_aic(
    int M,
    int N,
    int K,
    int vec_length,
    int K0,
    int N0,
    __gm__ half * __restrict__ values,
    __gm__ int * __restrict__ row_indices,
    __gm__ int * __restrict__ col_indices,
    __gm__ int * __restrict__ row_offsets,
    __gm__ half * __restrict__ B,
    __gm__ float * __restrict__ C)
{
    ascblasSpmmAIC(
        M,
        N,
        K,
        vec_length,
        K0,
        N0,
        values,
        row_indices,
        col_indices,
        row_offsets,
        B,
        C);
}
#endif
