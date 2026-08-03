#include "../../../utils.hpp"
#include "self_attention_cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// CPU 参考版本（src/ops/self_attention/cpu/self_attention.cpp）的算法说明，翻译 GPU-V1 时对照：
// V4: 两遍 softmax（第一遍算 score[j] 和 max_score，第二遍用 exp(score[j]-max_score) 做数值稳定的
// 加权求和），支持 GQA（nhead 是 nkvhead 的整数倍）。
// q        : [seqlen,    nhead,   d ]
// k        : [total_len, nkvhead, d ]
// qk^T     : [seqlen,    nhead,   total_len]
// softmax(scale * qk^T) : [seqlen, nhead, total_len]
// v        : [total_len, nkvhead, dv]
// attn_val : [seqlen,    nhead,   dv]
// attn_val = softmax(scale * q * k^T) * v

// 1. Kernel：只负责 GPU 上的具体计算
// GPU-V1：一个 block 处理一个 (query token i, head h)，网格设计跟 rope 一样用 blockIdx.x/y。
// scores 用动态 shared memory 存一整行（total_len 个 float），大小由 launcher 按
// total_len*sizeof(float) 申请，通过 <<<...>>> 第三个参数传入。
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
    // TODO：对照 CPU 版 self_attention_ 翻译成并行版本：
    //   1) 第一遍打分：每个线程按 grid-stride 负责一部分 j（j = tid, tid+stride, ... <= limit），
    //      算 scores[j] = scale * dot(q[i,h,:], k[j,kvh,:])，下标参考 CPU 版：
    //        q[i*nhead*d + h*d + dim]，k[j*nkvhead*d + kvh*d + dim]
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

    //   2) block 内做 max 规约得到 max_score（参考 rms_norm_cuda.cu 的 shared memory 树规约写法，
    //      这里规约的是 max 不是 sum，需要另一块 shared memory，或者复用 scores 规约后再放回去，自己设计）。
    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    max_score = sdata[0];

    //   3) 第二遍：每个线程把自己负责的 scores[j] 原地替换成 exp(scores[j] - max_score)，
    //      同时累加局部和，再做一次 block 内 sum 规约得到 sum_exp。

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
            sumexp[tid] = sumexp[tid]+ sumexp[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = sumexp[0];

    //   4) 第三遍：这次按 dv 维度分工（tid 覆盖 [0, dv) 而不是 [0, limit]），
    //      每个线程对负责的那个 t，遍历所有 j in [0, limit] 累加 scores[j] * v[j,kvh,t]，
    //      除以 sum_exp，写到 attn_val[i*nhead*dv + h*dv + t]。
    //   T 是 bf16/half 时记得转 float 参与运算，最后转回 T（参考 rope_kernel 的写法）。
    for (size_t t = tid; t < dv; t += blockDim.x) {
        float acc = 0.0;
        for (size_t j = 0; j <= limit; j += 1){
            acc += (float)(scores[j]) * (float)(v[j * nkvhead * dv + kvh * dv + t]);
        }
        attn_val[i * nhead * dv + h * dv + t] = acc / sum_exp;
        // attn_val[i * nhead * dv + h * dv + t] += acc[t] / sum_exp;
        // attn_val[i * nhead * dv + h * dv + t] = llaisys::utils::cast<T>(acc[t] / sum_exp);
    }
}

// 2. Launcher：负责 grid、block 和 kernel 启动
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

// 3. 对外接口：负责 std::byte 转换和 dtype 分发
namespace llaisys::ops::cuda {

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
