// src/device/nvidia/nvidia_resource.cu
// NVIDIA CUDA resource management

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

// CUDA resource structure
struct NvidiaResource {
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
    cudaStream_t stream;
};

// Global resource instance
static NvidiaResource g_nvidia_resource = {nullptr, nullptr, nullptr};

// Initialize CUDA resources
extern "C" llaisysResult_t llaisysNvidiaInit() {
    // Initialize cuBLAS
    cublasStatus_t cublas_status = cublasCreate(&g_nvidia_resource.cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        return LLAISYS_ERROR;
    }
    
    // Initialize cuDNN
    cudnnStatus_t cudnn_status = cudnnCreate(&g_nvidia_resource.cudnn_handle);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        cublasDestroy(g_nvidia_resource.cublas_handle);
        return LLAISYS_ERROR;
    }
    
    // Create CUDA stream
    cudaError_t cuda_status = cudaStreamCreate(&g_nvidia_resource.stream);
    if (cuda_status != cudaSuccess) {
        cudnnDestroy(g_nvidia_resource.cudnn_handle);
        cublasDestroy(g_nvidia_resource.cublas_handle);
        return LLAISYS_ERROR;
    }
    
    // Set stream for cuBLAS and cuDNN
    cublasSetStream(g_nvidia_resource.cublas_handle, g_nvidia_resource.stream);
    cudnnSetStream(g_nvidia_resource.cudnn_handle, g_nvidia_resource.stream);
    
    return LLAISYS_SUCCESS;
}

// Cleanup CUDA resources
extern "C" llaisysResult_t llaisysNvidiaCleanup() {
    if (g_nvidia_resource.stream) {
        cudaStreamDestroy(g_nvidia_resource.stream);
        g_nvidia_resource.stream = nullptr;
    }
    
    if (g_nvidia_resource.cudnn_handle) {
        cudnnDestroy(g_nvidia_resource.cudnn_handle);
        g_nvidia_resource.cudnn_handle = nullptr;
    }
    
    if (g_nvidia_resource.cublas_handle) {
        cublasDestroy(g_nvidia_resource.cublas_handle);
        g_nvidia_resource.cublas_handle = nullptr;
    }
    
    return LLAISYS_SUCCESS;
}

// Get cuBLAS handle
extern "C" cublasHandle_t llaisysNvidiaGetCublasHandle() {
    return g_nvidia_resource.cublas_handle;
}

// Get cuDNN handle
extern "C" cudnnHandle_t llaisysNvidiaGetCudnnHandle() {
    return g_nvidia_resource.cudnn_handle;
}

// Get CUDA stream
extern "C" cudaStream_t llaisysNvidiaGetStream() {
    return g_nvidia_resource.stream;
}
