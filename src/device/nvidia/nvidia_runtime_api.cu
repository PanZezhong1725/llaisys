#include "../runtime_api.hpp"

#include "cuda_check.cuh"

#include <cuda_runtime.h>

namespace llaisys::device::nvidia {
namespace runtime_api {

cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::runtime_error("Unsupported CUDA memcpy kind.");
    }
}

int getDeviceCount() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice) {
        cudaGetLastError();
        return 0;
    }
    CUDA_CHECK(status);
    return count;
}

void setDevice(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

void deviceSynchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
    }
}

void streamSynchronize(llaisysStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void *mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

void memcpySync(void *dst, const void *src, size_t size,
                llaisysMemcpyKind_t kind) {
    if (size == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size,
                 llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    if (size == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(
        dst, src, size, toCudaMemcpyKind(kind),
        reinterpret_cast<cudaStream_t>(stream)));
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
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
