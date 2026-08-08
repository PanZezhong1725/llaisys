// src/ops/add/nvidia/add_nvidia.cu
// NVIDIA CUDA implementation of add operator

#include "add_nvidia.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void add_kernel(const T *a, const T *b, T *c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Specialization for half precision
template <>
__global__ void add_kernel<__half>(const __half *a, const __half *b, __half *c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = __hadd(a[i], b[i]);
    }
}

// Specialization for bfloat16 precision
template <>
__global__ void add_kernel<__nv_bfloat16>(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = __hadd(a[i], b[i]);
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<float><<<gridSize, blockSize>>>(
            (const float *)a, (const float *)b, (float *)c, size);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<__half><<<gridSize, blockSize>>>(
            (const __half *)a, (const __half *)b, (__half *)c, size);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<__nv_bfloat16><<<gridSize, blockSize>>>(
            (const __nv_bfloat16 *)a, (const __nv_bfloat16 *)b, (__nv_bfloat16 *)c, size);
        break;
    case LLAISYS_DTYPE_F64:
        add_kernel<double><<<gridSize, blockSize>>>(
            (const double *)a, (const double *)b, (double *)c, size);
        break;
    case LLAISYS_DTYPE_I32:
        add_kernel<int32_t><<<gridSize, blockSize>>>(
            (const int32_t *)a, (const int32_t *)b, (int32_t *)c, size);
        break;
    case LLAISYS_DTYPE_I64:
        add_kernel<int64_t><<<gridSize, blockSize>>>(
            (const int64_t *)a, (const int64_t *)b, (int64_t *)c, size);
        break;
    default:
        std::cerr << "[ERROR] Unsupported data type for add: " << type << std::endl;
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

} // namespace llaisys::ops::nvidia
