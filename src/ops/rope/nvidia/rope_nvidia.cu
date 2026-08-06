// src/ops/rope/nvidia/rope_nvidia.cu
// NVIDIA CUDA implementation of RoPE operator

#include <cuda_runtime.h>
#include <math.h>
#include "llaisys/ops.h"
#include "llaisys/tensor.h"

// CUDA kernel for RoPE
__global__ void rope_kernel(
    const float *input,
    const int64_t *pos_ids,
    float *output,
    size_t seqlen,
    size_t nhead,
    size_t d,
    float theta
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seqlen * nhead * d / 2) {
        size_t seq_idx = i / (nhead * d / 2);
        size_t head_idx = (i / (d / 2)) % nhead;
        size_t d_idx = i % (d / 2);
        
        int64_t pos = pos_ids[seq_idx];
        float angle = pos / powf(theta, 2.0f * d_idx / d);
        
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        
        size_t a_idx = seq_idx * nhead * d + head_idx * d + d_idx;
        size_t b_idx = seq_idx * nhead * d + head_idx * d + d_idx + d / 2;
        
        float a = input[a_idx];
        float b = input[b_idx];
        
        output[a_idx] = a * cos_val - b * sin_val;
        output[b_idx] = b * cos_val + a * sin_val;
    }
}

// NVIDIA RoPE operator implementation
extern "C" llaisysResult_t llaisysRopeNvidia(
    llaisysTensor_t out,
    llaisysTensor_t in,
    llaisysTensor_t pos_ids,
    float theta
) {
    // Check input tensors
    if (!out || !in || !pos_ids) {
        return LLAISYS_ERROR;
    }
    
    // Get dimensions
    size_t seqlen = in->shape[0];
    size_t nhead = in->shape[1];
    size_t d = in->shape[2];
    
    // Get data pointers
    const float *in_data = (const float*)in->data;
    const int64_t *pos_data = (const int64_t*)pos_ids->data;
    float *out_data = (float*)out->data;
    
    // Calculate grid and block dimensions
    size_t numel = seqlen * nhead * d / 2;
    int blockSize = 256;
    int gridSize = (numel + blockSize - 1) / blockSize;
    
    // Launch kernel
    rope_kernel<<<gridSize, blockSize>>>(
        in_data, pos_data, out_data, seqlen, nhead, d, theta
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}
