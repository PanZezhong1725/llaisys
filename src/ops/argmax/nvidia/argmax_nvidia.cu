// src/ops/argmax/nvidia/argmax_nvidia.cu
// NVIDIA CUDA implementation of argmax operator

#include <cuda_runtime.h>
#include "llaisys/ops.h"
#include "llaisys/tensor.h"

// CUDA kernel for argmax
__global__ void argmax_kernel(const float *data, size_t n, int64_t *max_idx, float *max_val) {
    extern __shared__ float shared_data[];
    extern __shared__ int64_t shared_idx[];
    
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    shared_data[tid] = (i < n) ? data[i] : -INFINITY;
    shared_idx[tid] = i;
    __syncthreads();
    
    // Reduction to find maximum
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_data[tid] < shared_data[tid + s]) {
                shared_data[tid] = shared_data[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        max_idx[blockIdx.x] = shared_idx[0];
        max_val[blockIdx.x] = shared_data[0];
    }
}

// NVIDIA argmax operator implementation
extern "C" llaisysResult_t llaisysArgmaxNvidia(
    llaisysTensor_t max_idx,
    llaisysTensor_t max_val,
    llaisysTensor_t vals
) {
    // Check input tensors
    if (!max_idx || !max_val || !vals) {
        return LLAISYS_ERROR;
    }
    
    // Get data pointers
    const float *data = (const float*)vals->data;
    int64_t *idx_data = (int64_t*)max_idx->data;
    float *val_data = (float*)max_val->data;
    
    // Calculate grid and block dimensions
    size_t n = vals->numel;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Allocate temporary storage for block results
    int64_t *temp_idx;
    float *temp_val;
    cudaMalloc(&temp_idx, gridSize * sizeof(int64_t));
    cudaMalloc(&temp_val, gridSize * sizeof(float));
    
    // Launch kernel
    size_t shared_mem_size = blockSize * (sizeof(float) + sizeof(int64_t));
    argmax_kernel<<<gridSize, blockSize, shared_mem_size>>>(data, n, temp_idx, temp_val);
    
    // Copy results back
    cudaMemcpy(idx_data, temp_idx, sizeof(int64_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(val_data, temp_val, sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Free temporary storage
    cudaFree(temp_idx);
    cudaFree(temp_val);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}
