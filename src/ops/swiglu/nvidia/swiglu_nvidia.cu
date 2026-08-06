// src/ops/swiglu/nvidia/swiglu_nvidia.cu
// NVIDIA CUDA implementation of SwiGLU operator

#include <cuda_runtime.h>
#include "llaisys/ops.h"
#include "llaisys/tensor.h"

// CUDA kernel for SwiGLU
__global__ void swiglu_kernel(
    const float *gate,
    const float *up,
    float *out,
    size_t numel
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        float g = gate[i];
        float u = up[i];
        out[i] = u * g / (1.0f + expf(-g));
    }
}

// NVIDIA SwiGLU operator implementation
extern "C" llaisysResult_t llaisysSwigluNvidia(
    llaisysTensor_t out,
    llaisysTensor_t gate,
    llaisysTensor_t up
) {
    // Check input tensors
    if (!out || !gate || !up) {
        return LLAISYS_ERROR;
    }
    
    // Check tensor properties
    if (gate->numel != up->numel || gate->numel != out->numel) {
        return LLAISYS_ERROR;
    }
    
    // Get data pointers
    const float *gate_data = (const float*)gate->data;
    const float *up_data = (const float*)up->data;
    float *out_data = (float*)out->data;
    
    // Calculate grid and block dimensions
    size_t numel = gate->numel;
    int blockSize = 256;
    int gridSize = (numel + blockSize - 1) / blockSize;
    
    // Launch kernel
    swiglu_kernel<<<gridSize, blockSize>>>(gate_data, up_data, out_data, numel);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}
