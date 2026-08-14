#include "rms_norm_suda.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace llaisys::ops::suda {

template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight,
                                size_t hidden_size, float eps) {
    extern __shared__ float s_sumsq[];
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const T *row_in = in + static_cast<size_t>(row) * hidden_size;
    T *row_out = out + static_cast<size_t>(row) * hidden_size;

    float local = 0.0f;
    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(row_in[i]);
        local += v * v;
    }
    s_sumsq[tid] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_sumsq[0] = rsqrtf(s_sumsq[0] / static_cast<float>(hidden_size) + eps);
    }
    __syncthreads();

    float inv = s_sumsq[0];
    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(row_in[i]) * inv;
        float w = static_cast<float>(weight[i]);
        row_out[i] = static_cast<T>(v * w);
    }
}

template <typename T>
static void launch_rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
                            size_t seq_len, size_t hidden_size, float eps) {
    const int block = 256;
    rms_norm_kernel<T><<<static_cast<int>(seq_len), block, block * sizeof(float)>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight), hidden_size, eps);
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t seq_len, size_t hidden_size, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_rms_norm<float>(out, in, weight, seq_len, hidden_size, eps);
    case LLAISYS_DTYPE_F16:
        return launch_rms_norm<__half>(out, in, weight, seq_len, hidden_size, eps);
    case LLAISYS_DTYPE_BF16:
        return launch_rms_norm<__nv_bfloat16>(out, in, weight, seq_len, hidden_size, eps);
    default:
        break;
    }
}

} // namespace llaisys::ops::suda