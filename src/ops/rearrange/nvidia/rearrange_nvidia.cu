// src/ops/rearrange/nvidia/rearrange_nvidia.cu
#include "rearrange_nvidia.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

namespace llaisys::ops::nvidia {

void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, size_t numel) {
    size_t element_size;
    switch (type) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_I32:
    case LLAISYS_DTYPE_U32:
        element_size = 4;
        break;
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
    case LLAISYS_DTYPE_I16:
    case LLAISYS_DTYPE_U16:
        element_size = 2;
        break;
    case LLAISYS_DTYPE_F64:
    case LLAISYS_DTYPE_I64:
    case LLAISYS_DTYPE_U64:
        element_size = 8;
        break;
    case LLAISYS_DTYPE_I8:
    case LLAISYS_DTYPE_U8:
    case LLAISYS_DTYPE_BOOL:
    case LLAISYS_DTYPE_BYTE:
        element_size = 1;
        break;
    default:
        std::cerr << "[ERROR] Unsupported data type for rearrange: " << type << std::endl;
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaMemcpy(out, in, numel * element_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA memcpy failed");
    }
}

} // namespace llaisys::ops::nvidia
