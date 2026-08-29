// src/ops/swiglu/nvidia/swiglu_nvidia.cu
#include "swiglu_nvidia.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cmath>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void swiglu_kernel(const T *gate, const T *up, T *out, size_t numel) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        float g = static_cast<float>(gate[i]);
        float u = static_cast<float>(up[i]);
        // SiLU(gate) * up = gate * sigmoid(gate) * up
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        out[i] = static_cast<T>(g * sigmoid_g * u);
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    int blockSize = 256;
    int gridSize = (numel + blockSize - 1) / blockSize;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float><<<gridSize, blockSize>>>(
            (const float *)gate, (const float *)up, (float *)out, numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<__half><<<gridSize, blockSize>>>(
            (const __half *)gate, (const __half *)up, (__half *)out, numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<__nv_bfloat16><<<gridSize, blockSize>>>(
            (const __nv_bfloat16 *)gate, (const __nv_bfloat16 *)up, (__nv_bfloat16 *)out, numel);
        break;
    case LLAISYS_DTYPE_F64:
        swiglu_kernel<double><<<gridSize, blockSize>>>(
            (const double *)gate, (const double *)up, (double *)out, numel);
        break;
    default:
        std::cerr << "[ERROR] Unsupported data type for swiglu: " << type << std::endl;
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

} // namespace llaisys::ops::nvidia
