#include "rope_suda.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace llaisys::ops::suda {

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, float theta,
                            size_t seq_len, size_t num_heads, size_t head_dim) {
    size_t half_dim = head_dim / 2;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * num_heads * half_dim;
    if (idx >= total) {
        return;
    }

    size_t j = idx % half_dim;
    size_t h = (idx / half_dim) % num_heads;
    size_t i = idx / (half_dim * num_heads);

    size_t base_idx = (i * num_heads + h) * head_dim;

    float x1 = static_cast<float>(in[base_idx + j]);
    float x2 = static_cast<float>(in[base_idx + j + half_dim]);

    // freq = pos / (theta ** (2*j/head_dim))
    double exponent = 2.0 * static_cast<double>(j) / static_cast<double>(head_dim);
    double denom = pow(static_cast<double>(theta), exponent);
    double freq = static_cast<double>(pos_ids[i]) / denom;
    float c = static_cast<float>(cos(freq));
    float s = static_cast<float>(sin(freq));

    // Match Torch: x1 * cos - x2 * sin, x2 * cos + x1 * sin
    float out_first = x1 * c - x2 * s;
    float out_second = x2 * c + x1 * s;

    out[base_idx + j] = static_cast<T>(out_first);
    out[base_idx + j + half_dim] = static_cast<T>(out_second);
}

template <typename T>
static void launch_rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, float theta,
                        size_t seq_len, size_t num_heads, size_t head_dim) {
    size_t half_dim = head_dim / 2;
    size_t total = seq_len * num_heads * half_dim;
    const int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    rope_kernel<T><<<grid, block>>>(reinterpret_cast<T *>(out),
                                    reinterpret_cast<const T *>(in),
                                    pos_ids, theta,
                                    seq_len, num_heads, head_dim);
}

void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, float theta,
          llaisysDataType_t dtype, size_t seq_len, size_t num_heads, size_t head_dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_rope<float>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return launch_rope<__half>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return launch_rope<__nv_bfloat16>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
    default:
        break;
    }
}

} // namespace llaisys::ops::suda