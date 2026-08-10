#include "../../../utils.hpp"
#include "swiglu_cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// swiglu(gate, up) = silu(gate) * up，silu(x) = x/(1+exp(-x))。
// out/gate/up 都是 [rows,d] 且 contiguous（op.cpp 已断言），按 numel=rows*d 摊平成一维处理，
// 逐元素无跨元素依赖，不需要归约/shared memory。

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float gate_val = (float)gate[idx];
        gate_val = gate_val / (1 + std::exp(-gate_val)); // silu(gate)
        float up_val = (float)up[idx];
        float swiglu_val = gate_val * up_val;
        out[idx] = swiglu_val;
    }
}

template <typename T>
void launch_swiglu(T *out, const T *gate, const T *up,size_t rows,size_t d) {
    constexpr int block_size = 256;
    int numel = rows * d;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    swiglu_kernel<<<grid_size, block_size>>>(out, gate, up, numel);
}

namespace llaisys::ops::cuda {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t rows, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_swiglu(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            rows,
            d
        );
        return;

    case LLAISYS_DTYPE_BF16:
        launch_swiglu(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(gate),
            reinterpret_cast<const __nv_bfloat16 *>(up),
            rows,
            d
        );
        return;

    case LLAISYS_DTYPE_F16:
        launch_swiglu(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(gate),
            reinterpret_cast<const __half *>(up),
            rows,
            d
        );
        return;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}