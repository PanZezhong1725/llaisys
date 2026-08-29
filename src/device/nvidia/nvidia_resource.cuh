// src/device/nvidia/nvidia_resource.cuh
// NVIDIA CUDA / MetaX MACA resource management

#pragma once

#include "../device_resource.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace llaisys::device::nvidia {

class Resource : public DeviceResource {
private:
    cublasHandle_t _cublas_handle;
    cudaStream_t _stream;
    bool _initialized;

public:
    Resource(int device_id);
    ~Resource();

    // Prevent copying
    Resource(const Resource &) = delete;
    Resource &operator=(const Resource &) = delete;

    // Get CUDA resources
    cublasHandle_t cublasHandle() const { return _cublas_handle; }
    cudaStream_t stream() const { return _stream; }
    bool isInitialized() const { return _initialized; }

    // Initialize resources
    void init();
    void cleanup();
};

// Global resource accessor
Resource &getResource(int device_id = 0);

} // namespace llaisys::device::nvidia
