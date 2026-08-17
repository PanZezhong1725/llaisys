#include "../runtime_api.hpp"

#include "cuda_helpers.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace llaisys::device::nvidia {
namespace {

cudaMemcpyKind copyDirection(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D: return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return cudaMemcpyDeviceToDevice;
    default: throw std::invalid_argument("unknown memory copy direction");
    }
}

int deviceCount() {
    int count = 0;
    requireCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
}

void selectDevice(int device) { requireCuda(cudaSetDevice(device), "cudaSetDevice"); }
void synchronizeDevice() { requireCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

llaisysStream_t makeStream() {
    cudaStream_t stream = nullptr;
    requireCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void releaseStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        requireCuda(cudaStreamDestroy(cudaStream(stream)), "cudaStreamDestroy");
    }
}

void synchronizeStream(llaisysStream_t stream) {
    requireCuda(cudaStreamSynchronize(cudaStream(stream)), "cudaStreamSynchronize");
}

void *allocateDevice(size_t bytes) {
    if (bytes == 0) return nullptr;
    void *memory = nullptr;
    requireCuda(cudaMalloc(&memory, bytes), "cudaMalloc");
    return memory;
}

void releaseDevice(void *memory) {
    if (memory != nullptr) requireCuda(cudaFree(memory), "cudaFree");
}

void *allocateHost(size_t bytes) {
    if (bytes == 0) return nullptr;
    void *memory = nullptr;
    requireCuda(cudaMallocHost(&memory, bytes), "cudaMallocHost");
    return memory;
}

void releaseHost(void *memory) {
    if (memory != nullptr) requireCuda(cudaFreeHost(memory), "cudaFreeHost");
}

void copySync(void *destination, const void *source, size_t bytes, llaisysMemcpyKind_t kind) {
    if (bytes == 0) return;
    requireCuda(cudaMemcpy(destination, source, bytes, copyDirection(kind)), "cudaMemcpy");
}

void copyAsync(
    void *destination,
    const void *source,
    size_t bytes,
    llaisysMemcpyKind_t kind,
    llaisysStream_t stream) {
    if (bytes == 0) return;
    requireCuda(
        cudaMemcpyAsync(destination, source, bytes, copyDirection(kind), cudaStream(stream)),
        "cudaMemcpyAsync");
}

const LlaisysRuntimeAPI CUDA_API{
    &deviceCount,
    &selectDevice,
    &synchronizeDevice,
    &makeStream,
    &releaseStream,
    &synchronizeStream,
    &allocateDevice,
    &releaseDevice,
    &allocateHost,
    &releaseHost,
    &copySync,
    &copyAsync,
};

} // namespace

const LlaisysRuntimeAPI *getRuntimeAPI() { return &CUDA_API; }

} // namespace llaisys::device::nvidia
