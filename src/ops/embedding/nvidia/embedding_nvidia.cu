// src/ops/embedding/nvidia/embedding_nvidia.cu
#include "embedding_nvidia.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embedding_kernel(const int64_t *indices, const T *weight, T *output, size_t num_indices, size_t embedding_dim) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices) {
        int64_t idx = indices[i];
        for (size_t j = 0; j < embedding_dim; j++) {
            output[i * embedding_dim + j] = weight[idx * embedding_dim + j];
        }
    }
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t index_size, size_t weight_dim) {
    int blockSize = 256;
    int gridSize = (index_size + blockSize - 1) / blockSize;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float><<<gridSize, blockSize>>>((const int64_t *)index, (const float *)weight, (float *)out, index_size, weight_dim);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<__half><<<gridSize, blockSize>>>((const int64_t *)index, (const __half *)weight, (__half *)out, index_size, weight_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<__nv_bfloat16><<<gridSize, blockSize>>>((const int64_t *)index, (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)out, index_size, weight_dim);
        break;
    case LLAISYS_DTYPE_F64:
        embedding_kernel<double><<<gridSize, blockSize>>>((const int64_t *)index, (const double *)weight, (double *)out, index_size, weight_dim);
        break;
    default:
        std::cerr << "[ERROR] Unsupported data type for embedding: " << type << std::endl;
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

} // namespace llaisys::ops::nvidia
