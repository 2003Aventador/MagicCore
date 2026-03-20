
// spmm_kernel.cpp 与 spmm_kernel.cce文件内容完全相同，请忽略.cpp文件


#include "kernel_operator.h"
#include "ascblas_kernel_utils.h"
#include "stdio.h"

using namespace fp16;

#define NUM_ELE_PERBLOCK 16
#define CUBE_M0 16
#define CUBE_N0 16
#define CUBE_K0 16
#define CUBE_MATRIX_SIZE (16 * 16)

// 简单的向上取整宏
#define ROUND(v, n) ((v + n - 1) / n)

// 这个转置函数用来对矩阵的行列主序进行转换
// 要与昇腾文档中提供的“转置”概念区分开，昇腾提出的转置概念更像是重排布，主要是为了适配MMA计算的输入格式，而这个函数是真正意义上的矩阵转置
void transpose(__cbuf__ __fp16 *matrix, int M, int N) {
    int total = M * N;

    for (int start = 0; start < total; start++) {
        int current = start;

        // 找这个位置是否是循环的最小起点
        do {
            current = (current % M) * N + (current / M);
        } while (current > start);

        if (current < start) continue;

        // 开始做循环置换
        half temp = matrix[start];
        current = start;

        while (1) {
            int next = (current % M) * N + (current / M);
            if (next == start) break;

            matrix[current] = matrix[next];
            current = next;
        }

        matrix[current] = temp;
    }
}

__aicore__ __inline__ void ascblasSpmmAIC(
    int M,
    int N,
    int K,
    int vec_length,
    int K0,     // stride_K
    int N0,     // stride_N
    __gm__ half * __restrict__ values,
    __gm__ int * __restrict__ row_indices,          
    __gm__ int * __restrict__ col_indices,    
    __gm__ int * __restrict__ row_offsets,    // 这个变量用于后续负载均衡使用，暂时保留
    __gm__ half * __restrict__ B,
    __gm__ float * __restrict__ C
)
{
    auto L1_base_a = reinterpret_cast<__cbuf__ __fp16 *>((uintptr_t)0);            // 128 KB 128*256*2/1024=64 double buffer
    auto L1_base_b = reinterpret_cast<__cbuf__ __fp16 *>((uintptr_t)(128 * 1024)); // 128 KB 256*128
    auto L0A_base = reinterpret_cast<__ca__ __fp16 *>((uintptr_t)0); //128*128*2/1024=32 double buffer
    auto L0B_base = reinterpret_cast<__cb__ __fp16 *>((uintptr_t)0); //128*128*2/1024=32 double buffer
    auto L0C_base = reinterpret_cast<__cc__ float *>((uintptr_t)0); //128*128*4/1024=64 double buffer

    // 向量行数（向量块行数）
    int m_vec = M / vec_length;

    // 这两个循环变量暂时保留，后面扩展双缓冲的时候用
    int k0_ping_flag = 1; // 控制K0循环的双循环
    int n0_ping_flag = 1; // 控制N0循环的双循环

    // set_flag(PIPE_FIX, PIPE_M, EVENT_ID0); // 同步L0C的使用
    // set_flag(PIPE_FIX, PIPE_M, EVENT_ID1); // 同步L0C的使用
    // set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0); // 同步L1_buf_a的使用
    // set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1); // 同步L1_buf_a的使用
    // set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2); // 同步L1_buf_b的使用
    // set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3); // 同步L1_buf_b的使用
    // set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0); // 同步L1 -> (L0A | L0B)
    // set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1); // 同步L1 -> (L0A | L0B)

    // 还没搞明白同步机制
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);

    // 循环遍历每个行块
    for (int64_t vec_row = 0; vec_row < m_vec; vec_row++) {
        // 每个AICore负责一个行
        if (vec_row % get_block_num() != get_block_idx()) {
            continue;
        }
        // 计算该行的起始和结束位置（全局块索引）
        int row_start = row_indices[vec_row];
        int row_end = row_indices[vec_row+1];
        // 每一行的非零向量个数
        int nnz_vectors = row_end - row_start;

        // 在K方向上的分块数
        int k_steps = ROUND(nnz_vectors, K0);
        // 在N方向上的分块数
        int n_steps = ROUND(N, N0);
        // 后面非16对齐的时候用
        int residue = nnz_vectors % K0;
        // 索引数组
        int index[K0];

        // 如果该行没有非零块，跳过
        if (nnz_vectors == 0) {
            continue;
        }

        // 循环遍历K方向的每个分块
        for (int k_step = 0; k_step < k_steps; k_step++) {
            // 这次遍历的实际步长
            int cnt = min(nnz_vectors - k_step * K0, K0);
            // 存放列索引
            for(int i = 0; i < cnt; i++){
                // 计算全局索引
                int global_elem_index = row_start + k_step * K0 + i;
                index[i] = col_indices[global_elem_index];
            }

            // Gm -> L1
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
            // 把Gm中的A的一个 vec_length * K0 块放到L1中
            auto src_A = values + k_step * vec_length;
            auto dst_A = L1_base_a;
            copy_gm_to_cbuf(
                dst_A, 
                src_A, 
                static_cast<uint8_t>(0),            // sid
                static_cast<uint16_t>(cnt),         // nBurst
                static_cast<uint16_t>(vec_length),  // lenBurst
                static_cast<uint16_t>(0),           // srcGap?没搞懂
                static_cast<uint16_t>(0),           // dstGap
                PAD_NONE                            // padMode
            );

            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);

            // 对L1中的A进行转置以确保满足L0行排布需求
            transpose(dst_A, vec_length, cnt);
            // transpose是原地操作，需要等待完成后再进行后续读取
            pipe_barrier(PIPE_MTE2);

            // B矩阵在N方向上的分块
            for(int n_step = 0; n_step < n_steps; n_step++){
                // B矩阵在N方向上的实际处理列数
                int remain = min(N0, N - n_step * N0);
                // 每次循环都需要等待L0C就绪，确保不会覆盖未完成写回的数据
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
                auto dst_B = L1_base_b;
                // 一次拷贝B的一行，有多少索引拷贝多少行
                for(int i = 0; i < cnt; i++){
                    int cur_col_index = index[i];
                    auto src_B = B + cur_col_index * N + n_step * N0;
                    copy_gm_to_cbuf(
                        dst_B + i * remain,
                        src_B,
                        static_cast<uint8_t>(0),            // sid
                        static_cast<uint16_t>(1),           // nBurst
                        static_cast<uint16_t>(remain),      // lenBurst, TODO: padding
                        static_cast<uint16_t>(0),           // srcGap, 只拷贝B的一行，不需要使用该参数
                        static_cast<uint16_t>(0)            // dstGap, 只拷贝B的一行，不需要使用该参数
                    );
                }
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);

                // 对B进行转置以满足L0的列优先策略
                transpose(dst_B, cnt, remain);
                // transpose是原地操作，需要等待完成后再进行后续读取
                pipe_barrier(PIPE_MTE2);

                // L1->L0 & 计算
                // 拷贝的时候transpose设置为true，将原本ND格式的A、B重排布为满足MAD的zz、nz格式
                auto dst_L0_a = L0A_base;
                auto dst_L0_b = L0B_base;
                auto dst_L0_c = L0C_base + k_step * vec_length * K0;
                // half精度的分形固定大小为256B，即MAD尺寸矩阵所占空间，即16 *16 *2B
                int fractal_size = vec_length * 16;
                load_cbuf_to_ca(
                    dst_L0_a,
                    dst_A,
                    static_cast<uint16_t>(0),                               // baseIdx
                    static_cast<uint8_t>(vec_length * cnt / fractal_size),  // repeat
                    static_cast<uint16_t>(1),                               // srcStride
                    static_cast<uint16_t>(0),                               // dstGap
                    static_cast<uint8_t>(0),                                // sid
                    static_cast<bool>(true),                                // transpose, 这里的分形转置只为了适配后续的MMA计算，而且更适合叫重排布
                    0                                                       // addr_cal_mode
                );
                load_cbuf_to_cb(
                    dst_L0_b,
                    dst_B,
                    static_cast<uint16_t>(0),                           // baseIdx
                    static_cast<uint8_t>(cnt * remain / fractal_size),  // repeat
                    static_cast<uint16_t>(1),                           // srcStride
                    static_cast<uint16_t>(0),                           // dstGap
                    static_cast<uint8_t>(0),                            // sid
                    static_cast<bool>(true),                            // transpose, 这里的分形转置只为了适配后续的MMA计算
                    0                                                   // addr_cal_mode
                );

                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);

                // K0的第一个分块需要初始化C矩阵，后续分块则累加
                bool cmatrixInitVal = k_step == 0; 

                // 不在L0进行二次分块，而是直接把大矩阵送到MAD中，由硬件进行分块调度
                // 文档里没说不可以，因为MAD本身可接受的MNK很大
                mad(
                    dst_L0_c,
                    dst_L0_a,
                    dst_L0_b,
                    static_cast<uint16_t>(vec_length),   // m
                    static_cast<uint16_t>(cnt),          // k
                    static_cast<uint16_t>(remain),       // n
                    static_cast<uint8_t>(0),             // featOffset
                    static_cast<uint8_t>(0),             // smaskOffset
                    static_cast<uint8_t>(0),             // unitFlag
                    static_cast<bool>(0),                // kDirectionAlign
                    static_cast<bool>(0),                // isWeightOffset
                    static_cast<bool>(0),                // cmatrixSource
                    static_cast<bool>(cmatrixInitVal)    // cmatrixInitVal
                );

                pipe_barrier(PIPE_M);
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            }
        }

        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

        // 当前块行执行完毕，从L0C搬运回GM
        auto src_L0_C = L0C_base;
        auto dst_Gm_C = C + vec_row * vec_length * N;
        /**
         * config[0:15]: K0/16 = 8, 0000 0000 0000 1000
         * config[16:31]: 1 0000 0000 0000 0001, 单位为fractal_size(16*16)
         * config[32:47]: 256(16*16), 0000 0001 0000 0000
         * = 33685512
        */
        set_nd_para(33,685,512);
        copy_matrix_cc_to_gm(
            dst_Gm_C,
            src_L0_C,
            static_cast<uint8_t>(0),                // sid
            static_cast<uint16_t>(N),               // NSize
            static_cast<uint16_t>(vec_length),      // MSize
            static_cast<uint32_t>(N),               // dstStride_dst_D
            static_cast<uint16_t>(M),               // srcStride
            static_cast<uint8_t>(0),                // UnitFlagMode
            5'b00000,                               // QuantRPE
            3'b000,                                 // ReLUPRE
            static_cast<bool>(0),                   // channelSplit
            static_cast<bool>(1)                    // NZ2ND_EN
        );

    // wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    // wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    // wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    // wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    // wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    // wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    // wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    // wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
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
    __gm__ float * __restrict__ C
)
{
    
    // 初始设置
    // set_ffts_base_addr((uint64_t)ffts_addr);
    // set_padding(0);
    // set_atomic_none();
    // set_nd_para(0x1);

    // 计算C = A(sparse) × B(dense)
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
        C
    );
    pipe_barrier(PIPE_ALL);
}
#endif