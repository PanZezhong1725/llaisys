// src/ops/rms_norm/nvidia/rms_norm_nvidia.cu
// NVIDIA CUDA implementation of RMS normalization operator

#include <cuda_runtime.h>
#include "llaisys/ops.h"
#include "llaisys/tensor.h"

// CUDA kernel for RMS normalization
__global__ void rms_norm_kernel(
    const float *input,
    const float *weight,
    float *output,
    size_t batch_size,
    size_t feature_size,
    float eps
) {
    size_t batch_idx = blockIdx.x;
    size_t feature_idx = threadIdx.x;
    
    if (batch_idx < batch_size && feature_idx < feature_size) {
        // Calculate sum of squares
        float sum_sq = 0.0f;
        for (size_t i = 0; i < feature_size; i++) {
            float val = input[batch_idx * feature_size + i];
            sum_sq += val * val;
        }
        
        // Calculate RMS
        float rms = sqrtf(sum_sq / feature_size + eps);
        
        // Normalize
        output[batch_idx * feature_size + feature_idx] = 
            weight[feature_idx] * input[batch_idx * feature_size + feature_idx] / rms;
    }
}

// NVIDIA RMS norm operator implementation
extern "C" llaisysResult_t llaisysRmsNormNvidia(
    llaisysTensor_t out,
    llaisysTensor_t in,
    llaisysTensor_t weight,
    float eps
) {
    // Check input tensors
    if (!out || !in || !weight) {
        return LLAISYS_ERROR;
    }
    
    // Get dimensions
    size_t batch_size = in->shape[0];
    size_t feature_size = in->shape[1];
    
    // Get data pointers
    const float *in_data = (const float*)in->data;
    const float *weight_data = (const float*)weight->data;
    float *out_data = (float*)out->data;
    
    // Calculate grid and block dimensions
    dim3 blockSize(feature_size);
    dim3 gridSize(batch_size);
    
    // Launch kernel
    rms_norm_kernel<<<gridSize, blockSize>>>(
        in_data, weight_data, out_data, batch_size, feature_size, eps
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}
