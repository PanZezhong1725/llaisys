#include "../../../utils.hpp"
#include "flash_attention_cuda.cuh"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Split-KV / flash-decoding，phase 1（局部计算）。
// 跟 flash_attention_decode_cuda.cu 的区别：那边一个 warp 扫完整个 [0, total_len)；
// 这里把 [0, total_len) 切成 num_splits 段，一个 warp 只扫自己负责的 [split_start, split_end)，
// 算出局部的 (m, l, acc) 三元组写进中间 buffer，不做最终的 acc/l 归一化——归一化和跨 split
// 合并留给 phase 2（另一个 kernel，负责把同一个 head 的 num_splits 份局部结果 rescale 后加起来）。
//
// grid 组织：blockIdx.x = head，blockIdx.y = split_idx（对应 dim3 grid(nhead, num_splits)）。
//
// 中间 buffer 布局（建议，可按你的习惯调整，只要 phase 2 读的时候对得上）：
//   partial_m  : [nhead, num_splits]              局部 running max
//   partial_l  : [nhead, num_splits]              局部 sum_exp
//   partial_acc: [nhead, num_splits, HEAD_DIM]     局部加权和（未除以 l），dim 排布沿用
//                原 kernel 的 item*WARP_SIZE+lane 交错方式，phase 2 按同样方式读

namespace {

constexpr int WARP_SIZE = 32;
constexpr int HEAD_DIM = 128;
constexpr int TILE_K = 32;

// 跟 flash_attention_decode_cuda.cu 里的同名函数完全一样，这里保留一份是因为两个 .cu
// 文件都在各自的匿名 namespace 里（内部链接），不会有 ODR 重名问题。
__device__ float warp_max(float value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value = max(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return __shfl_sync(0xffffffff, value, 0);
}

__device__ float warp_sum(float value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return __shfl_sync(0xffffffff, value, 0);
}

// 作用是返回一部分kernel 的值
template <typename T>
__global__ void flash_attention_decode_splitkv_phase1_kernel(
    float *partial_m, float *partial_l, float *partial_acc,
    const T *q, const T *k, const T *v,
    size_t total_len, size_t nhead, size_t nkvhead, float scale,
    size_t split_size, size_t num_splits) {
    const int lane = threadIdx.x;
    const size_t head = blockIdx.x;
    const size_t split_idx = blockIdx.y;
    const size_t kv_head = head / (nhead / nkvhead);

    // TODO: 算出这个 split 负责的 KV 范围。
    size_t split_start = split_idx * split_size;
    size_t split_end = min(split_start + split_size, total_len);
    if (split_start >= total_len) {
        partial_m[head * num_splits + split_idx] = -INFINITY;
        partial_l[head * num_splits + split_idx] = 0;
        for (size_t t = 0; t < 4; t++) {
            partial_acc[(head * num_splits + split_idx) * HEAD_DIM + t * WARP_SIZE + lane] = 0;
        }
        return;
    }
    extern __shared__ float shared[];
    float *k_tile = shared;
    float *v_tile = shared + TILE_K * HEAD_DIM;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t tile_start = split_start; tile_start < split_end; tile_start += TILE_K) {
        const size_t tile_size = min(static_cast<size_t>(TILE_K), split_end - tile_start);

        // 32 个 lane 协作将当前 K/V tile 转为 float 后搬入 shared memory。
        for (size_t flat = lane; flat < tile_size * HEAD_DIM; flat += WARP_SIZE) {
            const size_t row = flat / HEAD_DIM;
            const size_t dim = flat % HEAD_DIM;
            k_tile[flat] = static_cast<float>(
                k[(tile_start + row) * nkvhead * HEAD_DIM + kv_head * HEAD_DIM + dim]);
            v_tile[flat] = static_cast<float>(
                v[(tile_start + row) * nkvhead * HEAD_DIM + kv_head * HEAD_DIM + dim]);
        }
        __syncwarp();

        float score = -INFINITY;
        if (lane < tile_size) {
            score = 0.0f;
            for (int dim = 0; dim < HEAD_DIM; ++dim) {
                score += static_cast<float>(q[head * HEAD_DIM + dim]) * k_tile[lane * HEAD_DIM + dim];
            }
            score *= scale;
        }

        // 合并当前 tile 与历史 tile 的 online softmax；max 变化时需重缩放历史累加值。
        const float tile_max = warp_max(score);
        const float new_max = max(max_score, tile_max);
        const float old_scale = expf(max_score - new_max);
        const float probability = lane < tile_size ? expf(score - new_max) : 0.0f;

        sum_exp = sum_exp * old_scale + warp_sum(probability);
        for (int item = 0; item < 4; ++item) {
            acc[item] *= old_scale;
        }

        // 每个 lane 持有一个 key 的权重，通过 shuffle 广播后完成 P*V 累加。
        for (int source = 0; source < TILE_K; ++source) {
            const float weight = __shfl_sync(0xffffffff, probability, source);
            if (source < tile_size) {
                for (int item = 0; item < 4; ++item) {
                    acc[item] += weight * v_tile[source * HEAD_DIM + item * WARP_SIZE + lane];
                }
            }
        }
        max_score = new_max;
        __syncwarp();
    }

    partial_m[head * num_splits + split_idx] = max_score;
    partial_l[head * num_splits + split_idx] = sum_exp;
    for (size_t item = 0; item < 4; item++) {
        partial_acc[(head * num_splits + split_idx) * HEAD_DIM + item * WARP_SIZE + lane] = acc[item];
    }
}

template <typename T>
void launch_flash_attention_decode_splitkv_phase1(
    float *partial_m, float *partial_l, float *partial_acc,
    const T *q, const T *k, const T *v,
    size_t total_len, size_t nhead, size_t nkvhead, float scale,
    size_t split_size, size_t num_splits) {
    const size_t shared_bytes = TILE_K * HEAD_DIM * 2 * sizeof(float);
    dim3 grid(static_cast<unsigned int>(nhead), static_cast<unsigned int>(num_splits));
    flash_attention_decode_splitkv_phase1_kernel<<<grid, WARP_SIZE, shared_bytes>>>(
        partial_m, partial_l, partial_acc, q, k, v,
        total_len, nhead, nkvhead, scale, split_size, num_splits);
}

// Phase 2：跨 split 归约。一个 block（一个 warp）处理一个 head，串行扫过该 head 的
// num_splits 份局部结果——num_splits 很小（个位数到几十），串行开销可忽略，且避免了
// 再开一次跨 block 的规约/同步。每个 lane 各自重复扫一遍算出同样的 final_max，是拿
// 冗余计算换掉一次 broadcast，逻辑更简单。
// rescale 公式与 tile 内 online softmax 完全一致：sum/acc 各自乘上 exp(m_split - m_final)
// 再累加；m 为 -INFINITY 的空 split（tile 循环里提前 return 那种）直接跳过，避免 exp(-inf-(-inf))=nan。
template <typename T>
__global__ void flash_attention_decode_splitkv_phase2_kernel(
    T *output, const float *partial_m, const float *partial_l, const float *partial_acc,
    size_t num_splits) {
    const int lane = threadIdx.x;
    const size_t head = blockIdx.x;

    float final_max = -INFINITY;
    for (size_t s = 0; s < num_splits; ++s) {
        final_max = max(final_max, partial_m[head * num_splits + s]);
    }

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float final_sum = 0.0f;
    for (size_t s = 0; s < num_splits; ++s) {
        const float m_s = partial_m[head * num_splits + s];
        if (m_s == -INFINITY) {
            continue;
        }
        const float rescale = expf(m_s - final_max);
        final_sum += partial_l[head * num_splits + s] * rescale;
        for (int item = 0; item < 4; ++item) {
            acc[item] += partial_acc[(head * num_splits + s) * HEAD_DIM + item * WARP_SIZE + lane] * rescale;
        }
    }

    for (int item = 0; item < 4; ++item) {
        output[head * HEAD_DIM + item * WARP_SIZE + lane] = static_cast<T>(acc[item] / final_sum);
    }
}

template <typename T>
void launch_flash_attention_decode_splitkv_phase2(
    T *output, const float *partial_m, const float *partial_l, const float *partial_acc,
    size_t nhead, size_t num_splits) {
    flash_attention_decode_splitkv_phase2_kernel<<<static_cast<unsigned int>(nhead), WARP_SIZE>>>(
        output, partial_m, partial_l, partial_acc, num_splits);
}

// 根据 total_len 选 num_splits：目标是让 nhead*num_splits 个 block 大致打满常见 GPU 的
// SM 数量级（kTargetBlocks），同时不能切得比 kMinSplitSize 还碎——太碎的话 phase1 里
// 大量 tail split 只有一两个 tile,收益被 kernel launch/phase2 归约开销吃掉。
size_t choose_num_splits(size_t total_len, size_t nhead) {
    constexpr size_t kTargetBlocks = 64;
    constexpr size_t kMinSplitSize = TILE_K;
    constexpr size_t kMaxNumSplits = 32;

    size_t num_splits = (kTargetBlocks + nhead - 1) / nhead;
    num_splits = std::max<size_t>(1, std::min(num_splits, kMaxNumSplits));

    size_t max_useful_splits = std::max<size_t>(1, total_len / kMinSplitSize);
    num_splits = std::min(num_splits, max_useful_splits);
    return num_splits;
}

template <typename T>
void launch_flash_attention_decode_splitkv(
    T *output, const T *q, const T *k, const T *v,
    size_t total_len, size_t nhead, size_t nkvhead, float scale) {
    const size_t num_splits = choose_num_splits(total_len, nhead);
    const size_t split_size = (total_len + num_splits - 1) / num_splits;

    float *partial_m = nullptr;
    float *partial_l = nullptr;
    float *partial_acc = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&partial_m), nhead * num_splits * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&partial_l), nhead * num_splits * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&partial_acc), nhead * num_splits * HEAD_DIM * sizeof(float));

    launch_flash_attention_decode_splitkv_phase1(
        partial_m, partial_l, partial_acc, q, k, v,
        total_len, nhead, nkvhead, scale, split_size, num_splits);
    launch_flash_attention_decode_splitkv_phase2(
        output, partial_m, partial_l, partial_acc, nhead, num_splits);

    cudaFree(partial_m);
    cudaFree(partial_l);
    cudaFree(partial_acc);
}

} // namespace

namespace llaisys::ops::cuda {

void flash_attention_decode_splitkv(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                                    llaisysDataType_t type,
                                    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv,
                                    float scale) {
    ASSERT(seqlen == 1 && d == HEAD_DIM && dv == HEAD_DIM,
           "flash_attention_decode_splitkv requires seqlen=1 and d=dv=128");

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_flash_attention_decode_splitkv(reinterpret_cast<float *>(attn_val),
                                              reinterpret_cast<const float *>(q),
                                              reinterpret_cast<const float *>(k),
                                              reinterpret_cast<const float *>(v),
                                              total_len, nhead, nkvhead, scale);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_flash_attention_decode_splitkv(reinterpret_cast<__nv_bfloat16 *>(attn_val),
                                              reinterpret_cast<const __nv_bfloat16 *>(q),
                                              reinterpret_cast<const __nv_bfloat16 *>(k),
                                              reinterpret_cast<const __nv_bfloat16 *>(v),
                                              total_len, nhead, nkvhead, scale);
        return;
    case LLAISYS_DTYPE_F16:
        launch_flash_attention_decode_splitkv(reinterpret_cast<__half *>(attn_val),
                                              reinterpret_cast<const __half *>(q),
                                              reinterpret_cast<const __half *>(k),
                                              reinterpret_cast<const __half *>(v),
                                              total_len, nhead, nkvhead, scale);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cuda
