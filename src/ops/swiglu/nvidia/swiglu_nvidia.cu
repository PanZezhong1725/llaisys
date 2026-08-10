#include "swiglu_nvidia.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    float g = static_cast<float>(gate[idx]);
    float u = static_cast<float>(up[idx]);
    // silu(g) = g / (1 + exp(-g))
    float silu_g = g / (1.0f + __expf(-g));
    out[idx] = static_cast<T>(silu_g * u);
}

template <typename T>
static void launch_swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
                          size_t seq_len, size_t hidden_size) {
    size_t numel = seq_len * hidden_size;
    const int block = 256;
    int grid = static_cast<int>((numel + block - 1) / block);
    swiglu_kernel<T><<<grid, block>>>(reinterpret_cast<T *>(out),
                                      reinterpret_cast<const T *>(gate),
                                      reinterpret_cast<const T *>(up), numel);
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t seq_len, size_t hidden_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_swiglu<float>(out, gate, up, seq_len, hidden_size);
    case LLAISYS_DTYPE_F16:
        return launch_swiglu<__half>(out, gate, up, seq_len, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return launch_swiglu<__nv_bfloat16>(out, gate, up, seq_len, hidden_size);
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
