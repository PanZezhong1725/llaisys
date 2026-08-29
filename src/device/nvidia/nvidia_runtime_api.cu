// src/device/nvidia/nvidia_runtime_api.cu
// NVIDIA CUDA / MetaX MACA Runtime API implementation

#include "../runtime_api.hpp"

#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

namespace llaisys::device::nvidia {

namespace runtime_api {

int getDeviceCount() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    return count;
}

void setDevice(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to set CUDA device");
    }
}

void deviceSynchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to synchronize CUDA device");
    }
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaStreamCreate failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to create CUDA stream");
    }
    return (llaisysStream_t)stream;
}

void destroyStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    cudaError_t err = cudaStreamDestroy((cudaStream_t)stream);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaStreamDestroy failed: " << cudaGetErrorString(err) << std::endl;
    }
}

void streamSynchronize(llaisysStream_t stream) {
    if (stream == nullptr) {
        deviceSynchronize();
        return;
    }
    cudaError_t err = cudaStreamSynchronize((cudaStream_t)stream);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to synchronize CUDA stream");
    }
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to allocate CUDA device memory");
    }
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaFree failed: " << cudaGetErrorString(err) << std::endl;
    }
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        // Fallback to regular malloc if pinned memory allocation fails
        ptr = std::malloc(size);
        if (ptr == nullptr) {
            std::cerr << "[ERROR] mallocHost failed" << std::endl;
            throw std::runtime_error("Failed to allocate host memory");
        }
    }
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    // Try to free as pinned memory first, fallback to regular free
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
        std::free(ptr);
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        cuda_kind = cudaMemcpyHostToHost;
        break;
    case LLAISYS_MEMCPY_H2D:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
    case LLAISYS_MEMCPY_D2H:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
    case LLAISYS_MEMCPY_D2D:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
    default:
        std::cerr << "[ERROR] Unknown memcpy kind: " << kind << std::endl;
        throw std::runtime_error("Unknown memcpy kind");
    }
    
    cudaError_t err = cudaMemcpy(dst, src, size, cuda_kind);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to copy memory");
    }
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        cuda_kind = cudaMemcpyHostToHost;
        break;
    case LLAISYS_MEMCPY_H2D:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
    case LLAISYS_MEMCPY_D2H:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
    case LLAISYS_MEMCPY_D2D:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
    default:
        std::cerr << "[ERROR] Unknown memcpy kind: " << kind << std::endl;
        throw std::runtime_error("Unknown memcpy kind");
    }
    
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cuda_kind, (cudaStream_t)stream);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMemcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to copy memory asynchronously");
    }
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync
};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::nvidia
