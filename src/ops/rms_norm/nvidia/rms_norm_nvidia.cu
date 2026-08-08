// src/ops/rms_norm/nvidia/rms_norm_nvidia.cu
#include "rms_norm_nvidia.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rms_norm_kernel(const T *input, const T *weight, T *output, size_t rows, size_t cols, float eps) {
    size_t row = blockIdx.x;
    if (row >= rows) return;

    // Calculate sum of squares using shared memory reduction
    extern __shared__ float shared_sum[];
    size_t tid = threadIdx.x;
    
    float local_sum = 0.0f;
    for (size_t i = tid; i < cols; i += blockDim.x) {
        float val = static_cast<float>(input[row * cols + i]);
        local_sum += val * val;
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduction
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Calculate RMS
    float rms = sqrtf(shared_sum[0] / cols + eps);
    
    // Normalize
    for (size_t i = tid; i < cols; i += blockDim.x) {
        float val = static_cast<float>(input[row * cols + i]);
        float w = static_cast<float>(weight[i]);
        output[row * cols + i] = static_cast<T>(w * val / rms);
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t rows, size_t cols) {
    int blockSize = 256;
    size_t shared_mem = blockSize * sizeof(float);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float><<<rows, blockSize, shared_mem>>>(
            (const float *)in, (const float *)weight, (float *)out, rows, cols, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<__half><<<rows, blockSize, shared_mem>>>(
            (const __half *)in, (const __half *)weight, (__half *)out, rows, cols, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<__nv_bfloat16><<<rows, blockSize, shared_mem>>>(
            (const __nv_bfloat16 *)in, (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)out, rows, cols, eps);
        break;
    case LLAISYS_DTYPE_F64:
        rms_norm_kernel<double><<<rows, blockSize, shared_mem>>>(
            (const double *)in, (const double *)weight, (double *)out, rows, cols, eps);
        break;
    default:
        std::cerr << "[ERROR] Unsupported data type for rms_norm: " << type << std::endl;
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

} // namespace llaisys::ops::nvidia
