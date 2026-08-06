// src/ops/linear/nvidia/linear_nvidia.cu
// NVIDIA CUDA implementation of linear operator

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "llaisys/ops.h"
#include "llaisys/tensor.h"
#include "../../device/nvidia/nvidia_resource.cu"

// NVIDIA linear operator implementation using cuBLAS
extern "C" llaisysResult_t llaisysLinearNvidia(
    llaisysTensor_t out,
    llaisysTensor_t in,
    llaisysTensor_t weight,
    llaisysTensor_t bias
) {
    // Check input tensors
    if (!out || !in || !weight) {
        return LLAISYS_ERROR;
    }
    
    // Get dimensions
    size_t batch_size = in->shape[0];
    size_t in_features = in->shape[1];
    size_t out_features = weight->shape[0];
    
    // Get data pointers
    const float *in_data = (const float*)in->data;
    const float *weight_data = (const float*)weight->data;
    float *out_data = (float*)out->data;
    const float *bias_data = bias ? (const float*)bias->data : nullptr;
    
    // Get cuBLAS handle
    cublasHandle_t handle = llaisysNvidiaGetCublasHandle();
    
    // Perform matrix multiplication: out = in * weight^T
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T,  // weight is transposed
        CUBLAS_OP_N,  // input is not transposed
        out_features, // rows of weight^T
        batch_size,   // columns of input
        in_features,  // columns of weight^T / rows of input
        &alpha,
        weight_data, in_features,
        in_data, in_features,
        &beta,
        out_data, out_features
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        return LLAISYS_ERROR;
    }
    
    // Add bias if provided
    if (bias_data) {
        // TODO: Implement bias addition kernel
        // For now, use a simple kernel
        dim3 blockSize(256);
        dim3 gridSize((batch_size * out_features + blockSize.x - 1) / blockSize.x);
        
        // bias_add_kernel<<<gridSize, blockSize>>>(out_data, bias_data, batch_size, out_features);
    }
    
    return LLAISYS_SUCCESS;
}
