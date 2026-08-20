#include "swiglu_iluvatar.cuh"


#include "../../../utils.hpp"
#include "../../../device/iluvatar/iluvatar_dtype.cuh"
#include "../../../device/iluvatar/iluvatar_utils.cuh"

#include <cuda_runtime.h>

namespace llaisys::ops::iluvatar {

namespace {

using llaisys::device::iluvatar::to_float;
using llaisys::device::iluvatar::from_float;

template <typename T>
__global__ void swiglu_kernel(
    T *out,
    const T *gate,
    const T *up,
    size_t numel
) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    const float g = to_float<T>(gate[i]);

    const float u = to_float<T>(up[i]);

    float silu;

    if (g >= 0.0f) {
        silu = g / (1.0f + expf(-g));
    } else {
        const float e = expf(g);
        silu = g * e / (1.0f + e);
    }

    out[i] = from_float<T>(u * silu);
}

} // namespace

void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t dtype,
    size_t numel,
    llaisysStream_t stream_
) {
    constexpr int BLOCK_SIZE = 256;

    if (numel == 0) {
        return;
    }

    const int grid_size = static_cast<int>((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);

    switch (dtype) {

    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(gate),
                reinterpret_cast<const float *>(up),
                numel
            );
        break;

    case LLAISYS_DTYPE_F16:
        swiglu_kernel<__half>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<__half *>(out),
                reinterpret_cast<const __half *>(gate),
                reinterpret_cast<const __half *>(up),
                numel
            );
        break;

    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<__nv_bfloat16>
            <<<grid_size, BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16 *>(out),
                reinterpret_cast<const __nv_bfloat16 *>(gate),
                reinterpret_cast<const __nv_bfloat16 *>(up),
                numel
            );
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    ILUVATAR_CUDA_KERNEL_CHECK();
}

} // namespace llaisys::ops::iluvatar