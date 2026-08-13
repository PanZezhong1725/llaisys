#include "add_suda.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace llaisys::ops::suda {

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
static void launch_add(std::byte *c, const std::byte *a, const std::byte *b, size_t numel) {
    const int block = 256;
    int grid = static_cast<int>((numel + block - 1) / block);
    add_kernel<T><<<grid, block>>>(reinterpret_cast<T *>(c),
                                   reinterpret_cast<const T *>(a),
                                   reinterpret_cast<const T *>(b),
                                   numel);
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_add<float>(c, a, b, numel);
    case LLAISYS_DTYPE_F16:
        return launch_add<__half>(c, a, b, numel);
    case LLAISYS_DTYPE_BF16:
        return launch_add<__nv_bfloat16>(c, a, b, numel);
    default:
        break;
    }
}

} // namespace llaisys::ops::suda