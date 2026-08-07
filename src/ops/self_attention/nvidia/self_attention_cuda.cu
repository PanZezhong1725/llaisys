#include "../../../utils.hpp"
#include "self_attention_cuda.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// 手写的两遍 softmax kernel（GQA-aware，一个 block 处理一个 (query token i, head h)）。
// 之前有一版用 cuDNN 的 cudnn_frontend SDPA Graph API 做 head 维度是 8 的倍数时的加速路径，
// 但每次调用都要重新 build graph（decode 阶段 total_len 每步都变，缓存不了），实测是解码阶段
// 的主要瓶颈；而缓存方案又要在 causal mask 下正确处理 padding，还牵扯到 cuDNN 版本兼容性问题
// （9.24.0 在 sm_120 上 SDPA 运行时崩溃）和 Iluvatar 平台完全没有 Graph API 的问题。综合评估后
// 认为 cuDNN 加速这条路对这个项目现阶段不合适，已经去掉，统一用这版手写 kernel。
// 一个 block 处理一个 (query token i, head h)；scores 用动态 shared memory 存一整行
// （total_len 个 float），大小由 launcher 按 total_len*sizeof(float) 申请。
template <typename T>
__global__ void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                      size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                                      size_t d, size_t dv, float scale) {
    size_t i = blockIdx.x; // query token, [0, seqlen)
    size_t h = blockIdx.y; // query head, [0, nhead)
    size_t tid = threadIdx.x;

    size_t group = nhead / nkvhead; // 每 group 个 query head 共享 1 个 kv head
    size_t kvh = h / group;         // 该 query head 对应的 kv head
    size_t causal_offset = total_len - seqlen;
    size_t limit = i + causal_offset; // 因果掩码: j 的范围是 [0, limit]（闭区间）

    extern __shared__ float scores[]; // 大小 total_len，只用到 [0, limit] 这一段
    __shared__ float sdata[256];
    // 第一遍打分：每个线程按 grid-stride 负责一部分 j（j = tid, tid+stride, ... <= limit）。
    float max_score = -INFINITY;
    // 遍历每一个key token
    for (size_t j = tid; j <= limit; j += blockDim.x) {
        float score = 0.0f;

        for (size_t dim = 0; dim < d; dim++) {
            score += float(q[i * nhead * d + h * d + dim]) * float(k[j * nkvhead * d + kvh * d + dim]);
        }
        score *= scale;
        scores[j] = score;
        max_score = std::fmaxf(max_score, score);
    }
    sdata[tid] = max_score;
    __syncthreads();

    // block 内做 max 规约得到 max_score。
    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    max_score = sdata[0];

    __shared__ float sumexp[256];
    sumexp[tid] = 0.0;
    // 第二遍:用 exp(score - max_score) 做数值稳定的 softmax 加权求和。
    for (size_t j = tid; j <= limit; j += blockDim.x) {
        scores[j] = std::exp(scores[j] - max_score);
        sumexp[tid] += scores[j];
    }
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sumexp[tid] = sumexp[tid] + sumexp[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = sumexp[0];

    // 第三遍：按 dv 维度分工，加权求和 v 得到最终输出。
    for (size_t t = tid; t < dv; t += blockDim.x) {
        float acc = 0.0;
        for (size_t j = 0; j <= limit; j += 1) {
            acc += (float)(scores[j]) * (float)(v[j * nkvhead * dv + kvh * dv + t]);
        }
        attn_val[i * nhead * dv + h * dv + t] = acc / sum_exp;
    }
}

// Launcher：负责 grid、block 和 kernel 启动
template <typename T>
void launch_self_attention(T *attn_val, const T *q, const T *k, const T *v,
                           size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                           size_t d, size_t dv, float scale) {
    constexpr int block_size = 256;
    dim3 grid(static_cast<unsigned int>(seqlen), static_cast<unsigned int>(nhead));
    size_t shared_bytes = total_len * sizeof(float);

    self_attention_kernel<<<grid, block_size, shared_bytes>>>(
        attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale);
}

namespace llaisys::ops::cuda {
// q        : [seqlen,    nhead,   d ]
// k        : [total_len, nkvhead, d ]
// qk^T     : [seqlen,    nhead,   total_len]
// softmax(scale * qk^T) : [seqlen, nhead, total_len]
// v        : [total_len, nkvhead, dv]
// attn_val : [seqlen,    nhead,   dv]
// attn_val = softmax(scale * q * k^T) * v
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type,
                    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv,
                    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_self_attention(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_self_attention(
            reinterpret_cast<__nv_bfloat16 *>(attn_val),
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    case LLAISYS_DTYPE_F16:
        launch_self_attention(
            reinterpret_cast<__half *>(attn_val),
            reinterpret_cast<const __half *>(q),
            reinterpret_cast<const __half *>(k),
            reinterpret_cast<const __half *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cuda
