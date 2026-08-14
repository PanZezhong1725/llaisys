#include "rope_suda.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

    // Compute cos/sin on device. NOTE: the Iluvatar CoreX (ivcore) backend
    // does not provide a working float64 (double) math path (its torch port
    // warns "Limited support for torch.double"), so use float32 here.
    // freq = pos / (theta ** (2*j/head_dim))
    float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
    float denom = powf(theta, exponent);
    float freq = static_cast<float>(pos_ids[i]) / denom;
    float c = cosf(freq);
    float s = sinf(freq);

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
        launch_rope<float>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        launch_rope<__half>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        launch_rope<__nv_bfloat16>(out, in, pos_ids, theta, seq_len, num_heads, head_dim);
        break;
    default:
        break;
    }
}

} // namespace llaisys::ops::suda