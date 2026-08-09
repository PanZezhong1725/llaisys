#include "nvidia_resource.cuh"

#include "cuda_check.cuh"

#include <memory>
#include <unordered_map>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id)
    : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id),
      _cublas(nullptr) {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUBLAS_CHECK(cublasCreate(&_cublas));
}

Resource::~Resource() {
    if (_cublas != nullptr) {
        cublasDestroy(_cublas);
        _cublas = nullptr;
    }
}

cublasHandle_t Resource::cublas(llaisysStream_t stream) {
    CUBLAS_CHECK(cublasSetStream(
        _cublas, reinterpret_cast<cudaStream_t>(stream)));
    return _cublas;
}

Resource &getResource(int device_id) {
    thread_local std::unordered_map<int, std::unique_ptr<Resource>> resources;
    auto &resource = resources[device_id];
    if (resource == nullptr) {
        resource = std::make_unique<Resource>(device_id);
    }
    return *resource;
}

} // namespace llaisys::device::nvidia
