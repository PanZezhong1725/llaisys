#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t error_ = (call);                                       \
        if (error_ != cudaSuccess) {                                       \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                         __FILE__, __LINE__, cudaGetErrorString(error_));   \
            std::abort();                                                  \
        }                                                                  \
    } while (0)
namespace llaisys::device::iluvatar {

namespace runtime_api {

static cudaMemcpyKind convertMemcpyKind(llaisysMemcpyKind_t kind) {
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
            std::abort();
    }
}


int getDeviceCount() {
    int count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    CHECK_CUDA(cudaSetDevice(device));
}

void deviceSynchronize() {
    CHECK_CUDA(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    CHECK_CUDA(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)));
}

void streamSynchronize(llaisysStream_t stream) {
    CHECK_CUDA(cudaStreamSynchronize(reinterpret_cast<cudaStream_t> (stream)));
}

void *mallocDevice(size_t size) {
    void * ptr =nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    CHECK_CUDA(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    void * ptr =nullptr;
    CHECK_CUDA(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    CHECK_CUDA(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CHECK_CUDA(cudaMemcpy(dst, src, size, convertMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    CHECK_CUDA(cudaMemcpyAsync(dst, src, size, convertMemcpyKind(kind), reinterpret_cast<cudaStream_t>(stream)));
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
