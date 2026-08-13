#include "linear_suda.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace llaisys::ops::suda {

// out = in @ weight^T + bias
//   out:   [B, N]
//   in:    [B, K]
//   weight: [N, K] (row-major)
//   bias:  [N] (optional)
template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                              size_t B, size_t K, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = B * N;
    if (idx >= total) {
        return;
    }
    size_t b = idx / N;
    size_t n = idx % N;

    const T *in_row = in + b * K;
    const T *weight_row = weight + n * K;

    float sum = 0.0f;
    for (size_t k = 0; k < K; ++k) {
        sum += static_cast<float>(in_row[k]) * static_cast<float>(weight_row[k]);
    }
    if (bias != nullptr) {
        sum += static_cast<float>(bias[n]);
    }
    out[idx] = static_cast<T>(sum);
}

template <typename T>
static void launch_linear(std::byte *out, const std::byte *in, const std::byte *weight,
                          const std::byte *bias, size_t B, size_t K, size_t N) {
    size_t total = B * N;
    const int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    linear_kernel<T><<<grid, block>>>(reinterpret_cast<T *>(out),
                                      reinterpret_cast<const T *>(in),
                                      reinterpret_cast<const T *>(weight),
                                      reinterpret_cast<const T *>(bias),
                                      B, K, N);
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t B, size_t K, size_t N) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_linear<float>(out, in, weight, bias, B, K, N);
    case LLAISYS_DTYPE_F16:
        return launch_linear<__half>(out, in, weight, bias, B, K, N);
    case LLAISYS_DTYPE_BF16:
        return launch_linear<__nv_bfloat16>(out, in, weight, bias, B, K, N);
    default:
        break;
    }
}

} // namespace llaisys::ops::suda