#include "rope_nvidia.hpp"

#include "../../../utils.hpp"
#include "../../../utils/cuda_check.hpp"

#include <cmath>
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

constexpr int THREAD_NUM = 256;

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const std::int64_t *pos_ids, 
    size_t seq_len, size_t n_heads, size_t head_dim, float theta) {

    const int half_dim = static_cast<int>(head_dim >> 1);   // 每 head 一半
    const int heads_per_token = static_cast<int>(n_heads);
    const int pairs_per_token = heads_per_token * half_dim; // 一个 token 的配对数

    const int total_pairs = static_cast<int>(seq_len) * pairs_per_token;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= total_pairs) return;                              

    const int j = p % half_dim;
    const int head = (p / half_dim) % heads_per_token;
    const int token = p / pairs_per_token;

    const int a_idx = token * (heads_per_token * static_cast<int>(head_dim))
                    + head * static_cast<int>(head_dim) + j;
    const int b_idx = a_idx + half_dim;

    const float pos = static_cast<float>(pos_ids[token]);
    const float freq = pos / powf(theta, (2.0f * j) / static_cast<float>(head_dim));
    const float cos_val = cosf(freq);
    const float sin_val = sinf(freq);

    const float a = static_cast<float>(in[a_idx]);
    const float b = static_cast<float>(in[b_idx]);
    out[a_idx] = static_cast<T>(a * cos_val - b * sin_val);
    out[b_idx] = static_cast<T>(b * cos_val + a * sin_val);
}

namespace llaisys::ops::nvidia {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, llaisysDataType_t type, float theta) {
    const auto &shape = out->shape();
    ASSERT(shape.size() == 3, "RoPE expects 3D tensor [seq_len, n_heads, head_dim]");
    size_t seq_len = shape[0];
    size_t n_heads = shape[1];
    size_t head_dim = shape[2];
    ASSERT(head_dim % 2 == 0, "Head dimension must be even for RoPE.");

    // 每个线程处理一个配对
    const size_t total_pairs = seq_len * n_heads * (head_dim / 2);
    const int blocks = static_cast<int>((total_pairs + THREAD_NUM - 1) / THREAD_NUM);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float><<<blocks, THREAD_NUM>>>(
            reinterpret_cast<float *>(out->data()),
            reinterpret_cast<const float *>(in->data()),
            reinterpret_cast<const std::int64_t *>(pos_ids->data()),
            seq_len, n_heads, head_dim,
            theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<__half><<<blocks, THREAD_NUM>>>(
            reinterpret_cast<__half *>(out->data()),
            reinterpret_cast<const __half *>(in->data()),
            reinterpret_cast<const std::int64_t *>(pos_ids->data()),
            seq_len, n_heads, head_dim,
            theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<__nv_bfloat16><<<blocks, THREAD_NUM>>>(
            reinterpret_cast<__nv_bfloat16 *>(out->data()),
            reinterpret_cast<const __nv_bfloat16 *>(in->data()),
            reinterpret_cast<const std::int64_t *>(pos_ids->data()),
            seq_len, n_heads, head_dim,
            theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}
}