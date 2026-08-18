#include "../../../utils.hpp"
#include "add_iluvatar.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx < numel){
        c[idx] = a[idx] + b[idx];
    }
}

namespace llaisys::ops::iluvatar {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<grid_size,block_size>>>(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), numel);
        return;
    case LLAISYS_DTYPE_BF16:
        add_kernel<<<grid_size,block_size>>>(reinterpret_cast<__nv_bfloat16 *>(c), reinterpret_cast<const __nv_bfloat16 *>(a),
                    reinterpret_cast<const __nv_bfloat16 *>(b), numel);
        return;
    case LLAISYS_DTYPE_F16:
        add_kernel<<<grid_size,block_size>>>(reinterpret_cast<__half *>(c), reinterpret_cast<const __half *>(a),
                    reinterpret_cast<const __half *>(b), numel);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} 