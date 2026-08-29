// src/device/nvidia/nvidia_resource.cu
// NVIDIA CUDA / MetaX MACA resource management implementation

#include "nvidia_resource.cuh"

#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <mutex>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id)
    : DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id),
      _cublas_handle(nullptr),
      _stream(nullptr),
      _initialized(false) {
}

Resource::~Resource() {
    cleanup();
}

void Resource::init() {
    if (_initialized) {
        return;
    }

    // Set device
    cudaError_t err = cudaSetDevice(getDeviceId());
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to set CUDA device");
    }

    // Create stream
    err = cudaStreamCreate(&_stream);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaStreamCreate failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to create CUDA stream");
    }

    // Create cuBLAS handle
    cublasStatus_t status = cublasCreate(&_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[ERROR] cublasCreate failed: " << status << std::endl;
        cudaStreamDestroy(_stream);
        throw std::runtime_error("Failed to create cuBLAS handle");
    }

    // Set stream for cuBLAS
    status = cublasSetStream(_cublas_handle, nullptr);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[ERROR] cublasSetStream failed: " << status << std::endl;
        cublasDestroy(_cublas_handle);
        cudaStreamDestroy(_stream);
        throw std::runtime_error("Failed to set cuBLAS stream");
    }

    _initialized = true;
}

void Resource::cleanup() {
    if (!_initialized) {
        return;
    }

    if (_cublas_handle != nullptr) {
        cublasDestroy(_cublas_handle);
        _cublas_handle = nullptr;
    }

    if (_stream != nullptr) {
        cudaStreamDestroy(_stream);
        _stream = nullptr;
    }

    _initialized = false;
}

// Global resource map
static std::unordered_map<int, Resource *> g_resources;
static std::mutex g_resources_mutex;

Resource &getResource(int device_id) {
    std::lock_guard<std::mutex> lock(g_resources_mutex);
    auto it = g_resources.find(device_id);
    if (it == g_resources.end()) {
        Resource *res = new Resource(device_id);
        res->init();
        g_resources[device_id] = res;
        return *res;
    }
    
    if (!it->second->isInitialized()) {
        it->second->init();
    }
    
    return *it->second;
}


} // namespace llaisys::device::nvidia
