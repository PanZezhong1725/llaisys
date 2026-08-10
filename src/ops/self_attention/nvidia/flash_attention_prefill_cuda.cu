#include "../../../utils.hpp"
#include "flash_attention_cuda.cuh"
#include <cassert>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define CEIL(a, b) (((a) + (b) - 1) / (b))
// q/k/v/attn_val 布局同 V1：q[i*nhead*d+h*d+dim] / k,v 用 total_len 而非 seqlen 做行数。
// attn_val = causal_softmax(scale * q * k^T) * v；GQA：kvh = h / (nhead/nkvhead)。
// 内部累加统一用 float，写回时转回 T。
//
// 只服务 prefill（op.cpp 按 seqlen>1 && total_len==seqlen && d==dv==128 分流到这里；
// decode 走 flash_attention_decode_cuda.cu，其余 shape 落回 V1 self_attention_cuda.cu）。
// total_len==seqlen 是因为 qwen2.cc 一次性处理整个 prompt（非 chunked prefill），
// 所以 causal_offset 恒为 0，limit 直接等于 i；assert(total_len==seqlen) 防止以后
// 调用方式变了却没人发现（release 下是空操作，仅 debug 生效）。
//
// Tiling：行方向 TILE_Q=8 行/block，一个 warp 管一行；列方向 K/V 按 Bc=32(=warpSize)
// 一组分块进 shared memory 给 tile 内所有行共享，chunk 大小=warp size 省掉一层跨步
// 循环。Bc×(d+dv)×4B=32KB（d=dv=128），在 48KB 默认上限内。acc[4] 寄存器数组硬编码
// dv=128（32 lane×4）。

// __shfl_down_sync 规约后只有 lane 0 有完整结果，末尾用 __shfl_sync 广播给全 warp，
// 因为调用方（chunk_max/sum_exp）需要全 warp 一致的值参与后续 rescale。
__device__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value = max(value, other);
    }
    return __shfl_sync(0xffffffff, value, 0);
}

__device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value = value + other;
    }
    return __shfl_sync(0xffffffff, value, 0);
}

constexpr int block_size = 256;
constexpr int MAX_WARPS = block_size / 32;
const int TILE_Q = 8;
constexpr int Bc = 32;
template <typename T>
__global__ void flash_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                      size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                                      size_t d, size_t dv, float scale) {

    size_t tile_start = blockIdx.x * TILE_Q;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5; // = row_in_tile
    size_t i = tile_start + warp_id; // 可能 ≥ seqlen（tile 不满）
    size_t h = blockIdx.y;
    size_t group = nhead / nkvhead;
    size_t kvh = h / group;
    assert(total_len == seqlen); // 只服务 prefill，见文件头
    size_t limit = i;
    // 协作搬运+__syncthreads() 要求 block 内循环次数一致，不能用各 warp 自己的 limit，
    // 要用 tile 内最大值。
    size_t tile_max_limit = min(tile_start + TILE_Q - 1, seqlen - 1);

    extern __shared__ float smem[];
    float *K_chunk = smem;
    float *V_chunk = smem + Bc * d;
    float m = -INFINITY;
    float l = 0.0f;
    float acc[4] = {0.0};
    for (int j = 0; j <= tile_max_limit; j += Bc) {
        // 协作搬运 K/V 进 shared memory，tile 内所有行共享。
        for (size_t flat = threadIdx.x; flat < Bc * d; flat += blockDim.x) {
            size_t dim = flat % d;
            size_t row_in_chunk = flat / d;
            if (j + row_in_chunk <= tile_max_limit) {
                K_chunk[flat] = k[(j + row_in_chunk) * nkvhead * d + kvh * d + dim];
                V_chunk[flat] = v[(j + row_in_chunk) * nkvhead * dv + kvh * dv + dim];
            }
        }
        __syncthreads();

        // i 可能 ≥ seqlen（tile 不满）：协作搬运不受影响，但打分/softmax/写回只对
        // 真实存在的行有意义，越界读写 q/attn_val 不安全，所以整段包进判断里。i 在
        // 同一 warp 内一致，不会打破 warp_reduce_* 要求全 32 lane 参与 shuffle 的前提。
        if (i < seqlen) {
            float score = 0.0;
            for (size_t dim = 0; dim < d; dim++) {
                score += float(q[i * nhead * d + h * d + dim]) * K_chunk[lane * d + dim];
            }
            score *= scale;
            if (j + lane > limit) {
                score = -INFINITY;
            }
            float chunk_max = warp_reduce_max(score);
            if (chunk_max > m) {
                float suofang = exp(m - chunk_max);
                l *= suofang;
                for (int t = 0; t < 4; t++) {
                    acc[t] *= suofang;
                }
                m = chunk_max;
            }
            float p = exp(score - m);
            float sum_exp = warp_reduce_sum(p);
            l += sum_exp;
            for (int src = 0; src < 32; src++) {
                float p_src = __shfl_sync(0xffffffff, p, src);
                for (int t = 0; t < 4; t++) {
                    acc[t] += p_src * V_chunk[src * dv + t * 32 + lane];
                }
            }
        }
        // 必须等所有线程读完 K_chunk/V_chunk 才能进入下一轮覆盖它们——否则
        // i>=seqlen 的 warp（直接跳过上面的 if）会抢先覆盖，跟还在读的 warp 形成
        // data race（只有 tile 跨多个 chunk 时才触发）。
        __syncthreads();
    }
    if (i < seqlen) {
        for (size_t t = 0; t < 4; t++) {
            attn_val[i * nhead * dv + h * dv + t * 32 + lane] = acc[t] / l;
        }
    }
}

// 命名故意跟 V1 的 self_attention_kernel/launch_self_attention 不同：两个 .cu 里
// 出现同名同签名的函数模板会被链接器当弱符号合并，运行时随机选中哪一份（见
// docs/FLASH_ATTENTION_DEBUG_LOG_ZH.md Bug 1）。
template <typename T>
void launch_flash_attention(T *attn_val, const T *q, const T *k, const T *v,
                           size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                           size_t d, size_t dv, float scale) {
    unsigned int gridx = CEIL(seqlen, TILE_Q);

    dim3 grid(static_cast<unsigned int>(gridx), static_cast<unsigned int>(nhead));
    size_t shared_bytes = Bc * (d + dv) * sizeof(float);

    flash_attention_kernel<<<grid, block_size, shared_bytes>>>(
        attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale);
}

namespace llaisys::ops::cuda {

void flash_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                     llaisysDataType_t type,
                     size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv,
                     float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_flash_attention(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_flash_attention(
            reinterpret_cast<__nv_bfloat16 *>(attn_val),
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    case LLAISYS_DTYPE_F16:
        launch_flash_attention(
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
