// src/device/nvidia/nvidia_runtime_api.cu
// NVIDIA CUDA Runtime API implementation

#include <cuda_runtime.h>
#include "llaisys/runtime.h"
#include <cstring>

// Memory management
static llaisysResult_t nvidia_malloc(void **ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

static llaisysResult_t nvidia_free(void *ptr) {
    cudaError_t err = cudaFree(ptr);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

// Memory copy operations
static llaisysResult_t nvidia_memcpy_h2d(void *dst, const void *src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

static llaisysResult_t nvidia_memcpy_d2h(void *dst, const void *src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

static llaisysResult_t nvidia_memcpy_d2d(void *dst, const void *src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

// Memory set
static llaisysResult_t nvidia_memset(void *ptr, int value, size_t size) {
    cudaError_t err = cudaMemset(ptr, value, size);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

// Synchronization
static llaisysResult_t nvidia_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

// Device management
static llaisysResult_t nvidia_set_device(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

static llaisysResult_t nvidia_get_device(int *device_id) {
    cudaError_t err = cudaGetDevice(device_id);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

static llaisysResult_t nvidia_get_device_count(int *count) {
    cudaError_t err = cudaGetDeviceCount(count);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

// Register API
extern "C" const LlaisysRuntimeAPI *llaisysGetNvidiaRuntimeAPI() {
    static LlaisysRuntimeAPI api = {
        .malloc = nvidia_malloc,
        .free = nvidia_free,
        .memcpy_h2d = nvidia_memcpy_h2d,
        .memcpy_d2h = nvidia_memcpy_d2h,
        .memcpy_d2d = nvidia_memcpy_d2d,
        .memset = nvidia_memset,
        .synchronize = nvidia_synchronize,
        .set_device = nvidia_set_device,
        .get_device = nvidia_get_device,
        .get_device_count = nvidia_get_device_count,
    };
    return &api;
}
