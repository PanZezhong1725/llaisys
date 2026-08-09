#include "../runtime_api.hpp"

#include "corex_check.cuh"

#include <cuda_runtime.h>

namespace llaisys::device::corex::runtime_api {

cudaMemcpyKind toCorexMemcpyKind(llaisysMemcpyKind_t kind) {
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
        throw std::runtime_error("Unsupported CoreX memcpy kind.");
    }
}

int getDeviceCount() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice) {
        cudaGetLastError();
        return 0;
    }
    COREX_CHECK(status);
    return count;
}

void setDevice(int device) {
    COREX_CHECK(cudaSetDevice(device));
}

void deviceSynchronize() {
    COREX_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    COREX_CHECK(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        COREX_CHECK(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
    }
}

void streamSynchronize(llaisysStream_t stream) {
    COREX_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *pointer = nullptr;
    COREX_CHECK(cudaMalloc(&pointer, size));
    return pointer;
}

void freeDevice(void *pointer) {
    if (pointer != nullptr) {
        COREX_CHECK(cudaFree(pointer));
    }
}

void *mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *pointer = nullptr;
    COREX_CHECK(cudaMallocHost(&pointer, size));
    return pointer;
}

void freeHost(void *pointer) {
    if (pointer != nullptr) {
        COREX_CHECK(cudaFreeHost(pointer));
    }
}

void memcpySync(void *dst, const void *src, size_t size,
                llaisysMemcpyKind_t kind) {
    if (size != 0) {
        COREX_CHECK(cudaMemcpy(dst, src, size, toCorexMemcpyKind(kind)));
    }
}

void memcpyAsync(void *dst, const void *src, size_t size,
                 llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    if (size != 0) {
        COREX_CHECK(cudaMemcpyAsync(
            dst, src, size, toCorexMemcpyKind(kind),
            reinterpret_cast<cudaStream_t>(stream)));
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
    &memcpyAsync};

} // namespace llaisys::device::corex::runtime_api

namespace llaisys::device::corex {

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::corex
