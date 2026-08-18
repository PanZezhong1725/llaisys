#include "../../../utils.hpp"
#include "flash_attention_cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int HEAD_DIM = 128;
constexpr int TILE_K = 32;

// Warp 规约后将 lane 0 的结果广播给所有 lane，便于后续统一更新 online softmax 状态。
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

template <typename T>
__global__ void flash_attention_decode_kernel(T *output, const T *q, const T *k, const T *v,
                                              size_t total_len, size_t nhead, size_t nkvhead,
                                              float scale) {
    // Decode 时 seqlen=1：一个 block 处理一个 query head，一个 warp 处理该 head 的全部 KV cache。
    const int lane = threadIdx.x;
    const size_t head = blockIdx.x;
    const size_t kv_head = head / (nhead / nkvhead); // GQA

    // K/V 各占 TILE_K * HEAD_DIM 个 float，每次只保留当前 KV tile。
    extern __shared__ float shared[];
    float *k_tile = shared;
    float *v_tile = shared + TILE_K * HEAD_DIM;

    float max_score = -INFINITY; // 已处理 tile 的最大 attention score
    float sum_exp = 0.0f;        // 以 max_score 为基准的 softmax 分母
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // 每个 lane 负责 4 个输出维度

    for (size_t tile_start = 0; tile_start < total_len; tile_start += TILE_K) {
        const size_t tile_size = min(static_cast<size_t>(TILE_K), total_len - tile_start);

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

    for (int item = 0; item < 4; ++item) {
        output[head * HEAD_DIM + item * WARP_SIZE + lane] = static_cast<T>(acc[item] / sum_exp);
    }
}

template <typename T>
void launch_flash_attention_decode(T *output, const T *q, const T *k, const T *v,
                                   size_t total_len, size_t nhead, size_t nkvhead, float scale) {
    const size_t shared_bytes = TILE_K * HEAD_DIM * 2 * sizeof(float);
    flash_attention_decode_kernel<<<static_cast<unsigned int>(nhead), WARP_SIZE, shared_bytes>>>(
        output, q, k, v, total_len, nhead, nkvhead, scale);
}

} // namespace

namespace llaisys::ops::cuda {

void flash_attention_decode(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                            llaisysDataType_t type,
                            size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv,
                            float scale) {
    ASSERT(seqlen == 1 && d == HEAD_DIM && dv == HEAD_DIM,
           "flash_attention_decode requires seqlen=1 and d=dv=128");

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_flash_attention_decode(reinterpret_cast<float *>(attn_val),
                                      reinterpret_cast<const float *>(q),
                                      reinterpret_cast<const float *>(k),
                                      reinterpret_cast<const float *>(v),
                                      total_len, nhead, nkvhead, scale);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_flash_attention_decode(reinterpret_cast<__nv_bfloat16 *>(attn_val),
                                      reinterpret_cast<const __nv_bfloat16 *>(q),
                                      reinterpret_cast<const __nv_bfloat16 *>(k),
                                      reinterpret_cast<const __nv_bfloat16 *>(v),
                                      total_len, nhead, nkvhead, scale);
        return;
    case LLAISYS_DTYPE_F16:
        launch_flash_attention_decode(reinterpret_cast<__half *>(attn_val),
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
