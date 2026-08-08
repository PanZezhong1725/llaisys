// src/ops/argmax/nvidia/argmax_nvidia.cu
// NVIDIA CUDA implementation of argmax operator

#include "argmax_nvidia.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cfloat>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void argmax_kernel(const T *data, size_t n, int64_t *max_idx, T *max_val) {
    extern __shared__ char shared_mem[];
    T *shared_val = (T *)shared_mem;
    int64_t *shared_idx = (int64_t *)(shared_val + blockDim.x);
    
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    shared_val[tid] = (i < n) ? data[i] : T(-FLT_MAX);
    shared_idx[tid] = i;
    __syncthreads();
    
    // Reduction to find maximum
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_val[tid] < shared_val[tid + s]) {
                shared_val[tid] = shared_val[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        max_idx[blockIdx.x] = shared_idx[0];
        max_val[blockIdx.x] = shared_val[0];
    }
}

// Second stage kernel to reduce block results
template <typename T>
__global__ void argmax_reduce_kernel(const int64_t *block_idx, const T *block_val, size_t num_blocks, int64_t *max_idx, T *max_val) {
    extern __shared__ char shared_mem[];
    T *shared_val = (T *)shared_mem;
    int64_t *shared_idx = (int64_t *)(shared_val + blockDim.x);

    size_t tid = threadIdx.x;

    // Grid-stride over block results so num_blocks > blockDim.x works
    float best = -FLT_MAX;
    int64_t best_idx = 0;
    for (size_t i = tid; i < num_blocks; i += blockDim.x) {
        float v = static_cast<float>(block_val[i]);
        if (v > best) {
            best = v;
            best_idx = block_idx[i];
        }
    }
    shared_val[tid] = static_cast<T>(best);
    shared_idx[tid] = best_idx;
    __syncthreads();

    // Reduction to find maximum
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_val[tid] < shared_val[tid + s]) {
                shared_val[tid] = shared_val[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }

    // Write final result
    if (tid == 0) {
        max_idx[0] = shared_idx[0];
        max_val[0] = shared_val[0];
    }
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    int blockSize = 256;
    int gridSize = (numel + blockSize - 1) / blockSize;
    
    // Allocate temporary storage for block results
    int64_t *temp_idx;
    void *temp_val;
    cudaMalloc(&temp_idx, gridSize * sizeof(int64_t));
    
    size_t val_size;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        val_size = sizeof(float);
        break;
    case LLAISYS_DTYPE_F16:
        val_size = sizeof(__half);
        break;
    case LLAISYS_DTYPE_BF16:
        val_size = sizeof(__nv_bfloat16);
        break;
    case LLAISYS_DTYPE_F64:
        val_size = sizeof(double);
        break;
    default:
        std::cerr << "[ERROR] Unsupported data type for argmax: " << type << std::endl;
        throw std::runtime_error("Unsupported data type");
    }
    
    cudaMalloc(&temp_val, gridSize * val_size);
    
    size_t shared_mem_size = blockSize * (val_size + sizeof(int64_t));
    
    // Launch first stage kernel
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<float><<<gridSize, blockSize, shared_mem_size>>>(
            (const float *)vals, numel, temp_idx, (float *)temp_val);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<__half><<<gridSize, blockSize, shared_mem_size>>>(
            (const __half *)vals, numel, temp_idx, (__half *)temp_val);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<__nv_bfloat16><<<gridSize, blockSize, shared_mem_size>>>(
            (const __nv_bfloat16 *)vals, numel, temp_idx, (__nv_bfloat16 *)temp_val);
        break;
    case LLAISYS_DTYPE_F64:
        argmax_kernel<double><<<gridSize, blockSize, shared_mem_size>>>(
            (const double *)vals, numel, temp_idx, (double *)temp_val);
        break;
    }
    
    // If only one block, copy result directly
    if (gridSize == 1) {
        cudaMemcpy(max_idx, temp_idx, sizeof(int64_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(max_val, temp_val, val_size, cudaMemcpyDeviceToDevice);
    } else {
        // Launch second stage reduction
        int reduce_blockSize = 256;
        int reduce_gridSize = 1;
        size_t reduce_shared_mem = reduce_blockSize * (val_size + sizeof(int64_t));
        
        switch (type) {
        case LLAISYS_DTYPE_F32:
            argmax_reduce_kernel<float><<<reduce_gridSize, reduce_blockSize, reduce_shared_mem>>>(
                temp_idx, (const float *)temp_val, gridSize, (int64_t *)max_idx, (float *)max_val);
            break;
        case LLAISYS_DTYPE_F16:
            argmax_reduce_kernel<__half><<<reduce_gridSize, reduce_blockSize, reduce_shared_mem>>>(
                temp_idx, (const __half *)temp_val, gridSize, (int64_t *)max_idx, (__half *)max_val);
            break;
        case LLAISYS_DTYPE_BF16:
            argmax_reduce_kernel<__nv_bfloat16><<<reduce_gridSize, reduce_blockSize, reduce_shared_mem>>>(
                temp_idx, (const __nv_bfloat16 *)temp_val, gridSize, (int64_t *)max_idx, (__nv_bfloat16 *)max_val);
            break;
        case LLAISYS_DTYPE_F64:
            argmax_reduce_kernel<double><<<reduce_gridSize, reduce_blockSize, reduce_shared_mem>>>(
                temp_idx, (const double *)temp_val, gridSize, (int64_t *)max_idx, (double *)max_val);
            break;
        }
    }
    
    // Free temporary storage
    cudaFree(temp_idx);
    cudaFree(temp_val);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

} // namespace llaisys::ops::nvidia
