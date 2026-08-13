#include "self_attention_suda.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace llaisys::ops::suda {

// Layout:
//   q, out: [qlen, nhead, head_dim]   -> ld_qo = nhead * head_dim
//   k, v:   [kvlen, nkvhead, head_dim] -> ld_kv = nkvhead * head_dim
// GQA: query head h maps to kv head (h / (nhead / nkvhead)).
template <typename T>
__global__ void self_attention_kernel(T *out, const T *q, const T *k, const T *v,
                                      size_t qlen, size_t kvlen, size_t nhead,
                                      size_t nkvhead, size_t head_dim, float scale) {
    extern __shared__ float s_scores[]; // blockDim.x * kvlen

    int tid = threadIdx.x;
    size_t global = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = qlen * nhead;
    if (global >= total) {
        return;
    }

    size_t h = global % nhead;
    size_t qi = global / nhead;

    size_t num_repeats = nhead / nkvhead;
    size_t src_kv_head = h / num_repeats;

    size_t ld_qo = nhead * head_dim;
    size_t ld_kv = nkvhead * head_dim;

    const T *q_pos = q + qi * ld_qo + h * head_dim;
    float *scores = s_scores + tid * kvlen;

    // Causal mask: position i attends to j <= i + (kvlen - qlen).
    size_t diag_shift = kvlen - qlen;
    size_t max_j = qi + diag_shift;

    // Compute scores + causal mask, and track the row max for stability.
    float local_max = -INFINITY;
    for (size_t j = 0; j < kvlen; ++j) {
        float s;
        if (j <= max_j) {
            const T *k_pos = k + j * ld_kv + src_kv_head * head_dim;
            s = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
                s += static_cast<float>(q_pos[d]) * static_cast<float>(k_pos[d]);
            }
            s *= scale;
        } else {
            s = -INFINITY;
        }
        scores[j] = s;
        if (s > local_max) {
            local_max = s;
        }
    }

    // exp and sum.
    float sum = 0.0f;
    for (size_t j = 0; j < kvlen; ++j) {
        float e = expf(scores[j] - local_max);
        scores[j] = e;
        sum += e;
    }

    // Weighted sum of V.
    for (size_t d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (size_t j = 0; j < kvlen; ++j) {
            const T *v_pos = v + j * ld_kv + src_kv_head * head_dim;
            acc += scores[j] * static_cast<float>(v_pos[d]);
        }
        out[qi * ld_qo + h * head_dim + d] = static_cast<T>(acc / sum);
    }
}

template <typename T>
static void launch_self_attention(std::byte *out, const std::byte *q, const std::byte *k,
                                  const std::byte *v, size_t qlen, size_t kvlen,
                                  size_t nhead, size_t nkvhead, size_t head_dim, float scale) {
    size_t total = qlen * nhead;
    const int block = 128;
    int grid = static_cast<int>((total + block - 1) / block);
    size_t shmem = static_cast<size_t>(block) * kvlen * sizeof(float);
    self_attention_kernel<T><<<grid, block, shmem>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        qlen, kvlen, nhead, nkvhead, head_dim, scale);
}

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t qlen, size_t kvlen, size_t nhead,
                    size_t nkvhead, size_t head_dim, float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_self_attention<float>(out, q, k, v, qlen, kvlen, nhead, nkvhead, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return launch_self_attention<__half>(out, q, k, v, qlen, kvlen, nhead, nkvhead, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return launch_self_attention<__nv_bfloat16>(out, q, k, v, qlen, kvlen, nhead, nkvhead, head_dim, scale);
    default:
        break;
    }
}

} // namespace llaisys::ops::suda