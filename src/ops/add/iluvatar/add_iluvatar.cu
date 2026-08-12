#include "add_iluvatar.cuh"
#include "../../../device/iluvatar/iluvatar_utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>


__global__ void add_f32_kernel(float *c, const float *a, const float *b, size_t numel) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < numel) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_f16_kernel(__half *c, const __half *a, const __half *b, size_t numel) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < numel) {
        float x = __half2float(a[i]);
        float y = __half2float(b[i]);

        c[i] = __float2half_rn(x + y);
    }
}

__global__ void add_bf16_kernel(__nv_bfloat16 *c, const __nv_bfloat16 *a, const __nv_bfloat16 *b, size_t numel) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < numel) {
        float x = __bfloat162float(a[i]);
        float y = __bfloat162float(b[i]);

        c[i] = __float2bfloat16_rn(x + y);
    }
}

namespace llaisys::ops::iluvatar {

void add(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream_
) {
    constexpr int BLOCK_SIZE = 256;

    int grid_size = static_cast<int>((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);

    switch (type) {

    case LLAISYS_DTYPE_F32:
        add_f32_kernel<<<
            grid_size,
            BLOCK_SIZE,
            0,
            stream
        >>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel
        );
        break;

    case LLAISYS_DTYPE_F16:
        add_f16_kernel<<<
            grid_size,
            BLOCK_SIZE,
            0,
            stream
        >>>(
            reinterpret_cast<__half *>(c),
            reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b),
            numel
        );
        break;

    case LLAISYS_DTYPE_BF16:
        add_bf16_kernel<<<
            grid_size,
            BLOCK_SIZE,
            0,
            stream
        >>>(
            reinterpret_cast<__nv_bfloat16 *>(c),
            reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b),
            numel
        );
        break;

    default:
        throw std::runtime_error(
            "Unsupported datatype in ILUVATAR add"
        );
    }

    ILUVATAR_CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::iluvatar

