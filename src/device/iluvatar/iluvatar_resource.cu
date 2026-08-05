#include "iluvatar_resource.cuh"

namespace llaisys::device::iluvatar {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    cudaSetDevice(device_id);
    cublasCreate(&_cublas_handle);
    cudnnCreate(&_cudnn_handle);
}

Resource::~Resource() {
    cublasDestroy(_cublas_handle);
    cudnnDestroy(_cudnn_handle);
}

DeviceResource *getDeviceResource(int device_id){
    return new Resource(device_id);
}

} // namespace llaisys::device::iluvatar
