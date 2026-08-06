// src/ops/rearrange/nvidia/rearrange_nvidia.cu
// NVIDIA CUDA implementation of rearrange operator

#include <cuda_runtime.h>
#include "llaisys/ops.h"
#include "llaisys/tensor.h"

// CUDA kernel for tensor rearrangement
__global__ void rearrange_kernel(
    const float *src,
    float *dst,
    const size_t *src_strides,
    const size_t *dst_strides,
    const size_t *shape,
    size_t ndim,
    size_t numel
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // Calculate source and destination indices
        size_t src_idx = 0;
        size_t dst_idx = 0;
        size_t temp = idx;
        
        for (size_t i = 0; i < ndim; i++) {
            size_t coord = temp % shape[i];
            temp /= shape[i];
            src_idx += coord * src_strides[i];
            dst_idx += coord * dst_strides[i];
        }
        
        dst[dst_idx] = src[src_idx];
    }
}

// NVIDIA rearrange operator implementation
extern "C" llaisysResult_t llaisysRearrangeNvidia(
    llaisysTensor_t out,
    llaisysTensor_t in
) {
    // Check input tensors
    if (!out || !in) {
        return LLAISYS_ERROR;
    }
    
    // Get data pointers
    const float *in_data = (const float*)in->data;
    float *out_data = (float*)out->data;
    
    // Calculate grid and block dimensions
    size_t numel = in->numel;
    int blockSize = 256;
    int gridSize = (numel + blockSize - 1) / blockSize;
    
    // Launch kernel
    rearrange_kernel<<<gridSize, blockSize>>>(
        in_data, out_data,
        in->strides, out->strides,
        in->shape, in->ndim, numel
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}
