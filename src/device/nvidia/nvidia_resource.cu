#include "nvidia_resource.cuh"
#include "nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <unordered_map>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {}


/*
 *   Linear 在 Qwen2 一层里就会调用很多次，不要反复创建库 context
 *   
 *   cuBLAS 官方说明 handle 表示 cuBLAS context，
 *   而且 context 与当前 CUDA device 绑定；
 *   多设备时需要分别维护对应 handle
 */
namespace {
struct NvidiaThreadResources {
    // （设备 ID, cuBLAS 句柄）
    std::unordered_map<int, cublasHandle_t> cublas_handles;

    ~NvidiaThreadResources() {
        for (auto &[device, handle] : cublas_handles) {
            cudaSetDevice(device);
            cublasDestroy(handle);
        }
    }
};

thread_local NvidiaThreadResources resources;

} // namespace

cublasHandle_t getCublasHandle() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));  // 获取当前线程绑定的 GPU ID

    auto it = resources.cublas_handles.find(device);
    if (it != resources.cublas_handles.end()) {
        return it->second;  // 缓存命中：如果之前创建过，直接返回
    }

    // 缓存未命中：首次使用，现场创建
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    resources.cublas_handles.emplace(device, handle);   // 存入哈希表，供下次直接使用

    return handle;
}

// TODO : 以后可以重构成：
// 
// Runtime
//  └── NvidiaResource
//       └── cublasHandle_t
// 

} // namespace llaisys::device::nvidia
