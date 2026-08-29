// src/ops/rope/nvidia/rope_nvidia.cu
#include "rope_nvidia.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cmath>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void rope_kernel(const T *input, const int64_t *pos_ids, T *output, size_t seqlen, size_t nhead, size_t d, float theta) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t half_d = d / 2;
    size_t total = seqlen * nhead * half_d;
    
    if (i < total) {
        size_t seq_idx = i / (nhead * half_d);
        size_t head_idx = (i / half_d) % nhead;
        size_t d_idx = i % half_d;
        
        int64_t pos = pos_ids[seq_idx];
        float angle = static_cast<float>(pos) / powf(theta, 2.0f * d_idx / d);
        
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        size_t a_idx = seq_idx * nhead * d + head_idx * d + d_idx;
        size_t b_idx = seq_idx * nhead * d + head_idx * d + d_idx + half_d;
        
        float a = static_cast<float>(input[a_idx]);
        float b = static_cast<float>(input[b_idx]);
        
        output[a_idx] = static_cast<T>(a * cos_val - b * sin_val);
        output[b_idx] = static_cast<T>(b * cos_val + a * sin_val);
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d) {
    size_t total = seqlen * nhead * d / 2;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float><<<gridSize, blockSize>>>(
            (const float *)in, (const int64_t *)pos_ids, (float *)out, seqlen, nhead, d, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<__half><<<gridSize, blockSize>>>(
            (const __half *)in, (const int64_t *)pos_ids, (__half *)out, seqlen, nhead, d, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<__nv_bfloat16><<<gridSize, blockSize>>>(
            (const __nv_bfloat16 *)in, (const int64_t *)pos_ids, (__nv_bfloat16 *)out, seqlen, nhead, d, theta);
        break;
    case LLAISYS_DTYPE_F64:
        rope_kernel<double><<<gridSize, blockSize>>>(
            (const double *)in, (const int64_t *)pos_ids, (double *)out, seqlen, nhead, d, theta);
        break;
    default:
        std::cerr << "[ERROR] Unsupported data type for rope: " << type << std::endl;
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

} // namespace llaisys::ops::nvidia
