#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace {
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

cudaMemcpyKind to_cuda_kind(llaisysMemcpyKind_t kind) {
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
        throw std::invalid_argument("Invalid memcpy kind");
    }
}
} // namespace

namespace llaisys::device::iluvatar {
namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device_id) {
    check_cuda(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    check_cuda(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        check_cuda(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
    }
}

void streamSynchronize(llaisysStream_t stream) {
    check_cuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        check_cuda(cudaFree(ptr));
    }
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    check_cuda(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        check_cuda(cudaFreeHost(ptr));
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    check_cuda(cudaMemcpy(dst, src, size, to_cuda_kind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    check_cuda(cudaMemcpyAsync(dst, src, size, to_cuda_kind(kind), reinterpret_cast<cudaStream_t>(stream)));
}

const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount, &setDevice, &deviceSynchronize, &createStream, &destroyStream, &streamSynchronize,
    &mallocDevice, &freeDevice, &mallocHost, &freeHost, &memcpySync, &memcpyAsync};
} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::iluvatar
