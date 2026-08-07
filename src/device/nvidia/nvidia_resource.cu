#include "nvidia_resource.cuh"

namespace llaisys::device::nvidia {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    cudaSetDevice(device_id);
    cublasCreate(&_cublas_handle);
}

Resource::~Resource() {
    cublasDestroy(_cublas_handle);
}

DeviceResource *getDeviceResource(int device_id){
    return new Resource(device_id);
}

} // namespace llaisys::device::nvidia
