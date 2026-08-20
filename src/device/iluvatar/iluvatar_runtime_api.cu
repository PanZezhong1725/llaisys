#include "../runtime_api.hpp"
#include "iluvatar_utils.cuh"

#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

namespace llaisys::device::iluvatar {

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
        throw std::invalid_argument("Invalid memcpy kind");
    }
}

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    ILUVATAR_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    ILUVATAR_CUDA_CHECK(cudaSetDevice(device));
}

void deviceSynchronize() {
    ILUVATAR_CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    ILUVATAR_CUDA_CHECK(cudaStreamCreate(&stream));

    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    ILUVATAR_CUDA_CHECK(
        cudaStreamDestroy(
            reinterpret_cast<cudaStream_t>(stream)
        )
    );
}

void streamSynchronize(llaisysStream_t stream) {
    ILUVATAR_CUDA_CHECK(
        cudaStreamSynchronize(
            reinterpret_cast<cudaStream_t>(stream)
        )
    );
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    ILUVATAR_CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        ILUVATAR_CUDA_CHECK(cudaFree(ptr));
    }
}

// 得到 pinned/page-locked host memory，
// 后面使用 cudaMemcpyAsync() 时 host/device 之间的数据传输才能更好地配合异步执行
void *mallocHost(size_t size) {
    void *ptr = nullptr;
    ILUVATAR_CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        ILUVATAR_CUDA_CHECK(cudaFreeHost(ptr));
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    ILUVATAR_CUDA_CHECK(
        cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind))
    );
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    ILUVATAR_CUDA_CHECK(
        cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind),  reinterpret_cast<cudaStream_t>(stream))
    );
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
} // namespace llaisys::device::iluvatar
