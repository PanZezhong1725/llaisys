#include "../runtime_api.hpp"

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

namespace runtime_api {
void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

int getDeviceCount() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice) {
        cudaGetLastError();
        return 0;
    }
    check(status, "cudaGetDeviceCount");
    return count;
}

void setDevice(int device) {
    check(cudaSetDevice(device), "cudaSetDevice");
}

void deviceSynchronize() {
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        check(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamDestroy");
    }
}
void streamSynchronize(llaisysStream_t stream) {
    check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    check(cudaMalloc(&ptr, size), "cudaMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        check(cudaFree(ptr), "cudaFree");
    }
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    check(cudaMallocHost(&ptr, size), "cudaMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        check(cudaFreeHost(ptr), "cudaFreeHost");
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: cuda_kind = cudaMemcpyHostToHost; break;
    case LLAISYS_MEMCPY_H2D: cuda_kind = cudaMemcpyHostToDevice; break;
    case LLAISYS_MEMCPY_D2H: cuda_kind = cudaMemcpyDeviceToHost; break;
    case LLAISYS_MEMCPY_D2D: cuda_kind = cudaMemcpyDeviceToDevice; break;
    default: throw std::invalid_argument("invalid memcpy kind");
    }
    check(cudaMemcpy(dst, src, size, cuda_kind), "cudaMemcpy");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: cuda_kind = cudaMemcpyHostToHost; break;
    case LLAISYS_MEMCPY_H2D: cuda_kind = cudaMemcpyHostToDevice; break;
    case LLAISYS_MEMCPY_D2H: cuda_kind = cudaMemcpyDeviceToHost; break;
    case LLAISYS_MEMCPY_D2D: cuda_kind = cudaMemcpyDeviceToDevice; break;
    default: throw std::invalid_argument("invalid memcpy kind");
    }
    check(cudaMemcpyAsync(dst, src, size, cuda_kind, reinterpret_cast<cudaStream_t>(stream)), "cudaMemcpyAsync");
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
