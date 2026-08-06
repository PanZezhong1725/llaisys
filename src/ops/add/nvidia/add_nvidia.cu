// src/ops/add/nvidia/add_nvidia.cu
// NVIDIA CUDA implementation of add operator

#include <cuda_runtime.h>
#include "llaisys/ops.h"
#include "llaisys/tensor.h"

// CUDA kernel for element-wise addition
__global__ void add_kernel(const float *a, const float *b, float *c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// NVIDIA add operator implementation
extern "C" llaisysResult_t llaisysAddNvidia(
    llaisysTensor_t out,
    llaisysTensor_t a,
    llaisysTensor_t b
) {
    // Check input tensors
    if (!out || !a || !b) {
        return LLAISYS_ERROR;
    }
    
    // Check tensor properties
    if (a->numel != b->numel || a->numel != out->numel) {
        return LLAISYS_ERROR;
    }
    
    // Get data pointers
    const float *a_data = (const float*)a->data;
    const float *b_data = (const float*)b->data;
    float *out_data = (float*)out->data;
    
    // Calculate grid and block dimensions
    size_t n = a->numel;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch kernel
    add_kernel<<<gridSize, blockSize>>>(a_data, b_data, out_data, n);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}
