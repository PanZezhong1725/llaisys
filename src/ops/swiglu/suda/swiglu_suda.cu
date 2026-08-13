#include "swiglu_suda.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace llaisys::ops::suda {

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    float g = static_cast<float>(gate[idx]);
    float u = static_cast<float>(up[idx]);
    // silu(g) = g / (1 + exp(-g))
    float silu_g = g / (1.0f + expf(-g));
    out[idx] = static_cast<T>(silu_g * u);
}

template <typename T>
static void launch_swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
                          size_t numel) {
    const int block = 256;
    int grid = static_cast<int>((numel + block - 1) / block);
    swiglu_kernel<T><<<grid, block>>>(reinterpret_cast<T *>(out),
                                      reinterpret_cast<const T *>(gate),
                                      reinterpret_cast<const T *>(up), numel);
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_swiglu<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return launch_swiglu<__half>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return launch_swiglu<__nv_bfloat16>(out, gate, up, numel);
    default:
        break;
    }
}

} // namespace llaisys::ops::suda