// src/ops/embedding/nvidia/embedding_nvidia.cu
// NVIDIA CUDA implementation of embedding operator

#include <cuda_runtime.h>
#include "llaisys/ops.h"
#include "llaisys/tensor.h"

// CUDA kernel for embedding lookup
__global__ void embedding_kernel(
    const int64_t *indices,
    const float *weight,
    float *output,
    size_t num_indices,
    size_t embedding_dim
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_indices) {
        int64_t idx = indices[i];
        for (size_t j = 0; j < embedding_dim; j++) {
            output[i * embedding_dim + j] = weight[idx * embedding_dim + j];
        }
    }
}

// NVIDIA embedding operator implementation
extern "C" llaisysResult_t llaisysEmbeddingNvidia(
    llaisysTensor_t out,
    llaisysTensor_t index,
    llaisysTensor_t weight
) {
    // Check input tensors
    if (!out || !index || !weight) {
        return LLAISYS_ERROR;
    }
    
    // Get data pointers
    const int64_t *indices = (const int64_t*)index->data;
    const float *weight_data = (const float*)weight->data;
    float *out_data = (float*)out->data;
    
    // Calculate dimensions
    size_t num_indices = index->numel;
    size_t embedding_dim = weight->shape[1];
    
    // Calculate grid and block dimensions
    int blockSize = 256;
    int gridSize = (num_indices + blockSize - 1) / blockSize;
    
    // Launch kernel
    embedding_kernel<<<gridSize, blockSize>>>(
        indices, weight_data, out_data, num_indices, embedding_dim
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}
