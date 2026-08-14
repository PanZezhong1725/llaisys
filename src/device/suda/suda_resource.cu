#include "suda_resource.cuh"

#include <cuda_runtime.h>

namespace llaisys::device::suda {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_SUDA, device_id) {}

Resource::~Resource() {}

namespace {
// Thread-local cuBLAS handle. Each thread owns its own handle so that it can be
// used concurrently with the thread-local Context. The handle is created lazily
// on the currently active device and destroyed when the thread exits.
struct CublasHandleHolder {
    cublasHandle_t handle = nullptr;
    CublasHandleHolder() {
        cublasCreate(&handle);
    }
    ~CublasHandleHolder() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
    }
};

thread_local CublasHandleHolder tls_cublas_handle;
} // namespace

cublasHandle_t getCublasHandle() {
    return tls_cublas_handle.handle;
}

} // namespace llaisys::device::suda