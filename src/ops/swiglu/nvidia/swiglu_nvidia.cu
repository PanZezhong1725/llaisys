#include "swiglu_nvidia.hpp"

#include "../../../utils.hpp"
#include "../../../utils/cuda_check.hpp"

#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

constexpr int THREAD_NUM = 256;

template <typename T>
__global__ void swiglu_kernel(T *out,const T *gate, const T *up, size_t numel) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        float x = static_cast<float>(gate[index]);
        float sigmoid = 1 / (1 + exp(-x));
        out[index] = static_cast<T>(static_cast<float>(up[index]) * x * sigmoid);
    }
}

namespace llaisys::ops::nvidia {
void swiglu(tensor_t out, tensor_t gate, tensor_t up, llaisysDataType_t type, size_t numel) {
    const int blocks = static_cast<int>((numel + THREAD_NUM - 1) / THREAD_NUM);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float><<<blocks, THREAD_NUM>>>(
            reinterpret_cast<float *>(out->data()), 
            reinterpret_cast<const float *>(gate->data()),
            reinterpret_cast<const float *>(up->data()), numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<__half><<<blocks, THREAD_NUM>>>(
            reinterpret_cast<__half *>(out->data()), 
            reinterpret_cast<const __half *>(gate->data()),
            reinterpret_cast<const __half *>(up->data()), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<__nv_bfloat16><<<blocks, THREAD_NUM>>>(
            reinterpret_cast<__nv_bfloat16 *>(out->data()), 
            reinterpret_cast<const __nv_bfloat16 *>(gate->data()),
            reinterpret_cast<const __nv_bfloat16 *>(up->data()), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}
}