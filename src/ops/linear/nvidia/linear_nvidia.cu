// src/ops/linear/nvidia/linear_nvidia.cu
#include "linear_nvidia.cuh"
#include "../../../device/nvidia/nvidia_resource.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <iostream>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void bias_add_kernel(T *out, const T *bias, size_t batch, size_t out_features) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * out_features) {
        size_t col = i % out_features;
        out[i] = out[i] + bias[col];
    }
}

template <>
__global__ void bias_add_kernel<__half>(__half *out, const __half *bias, size_t batch, size_t out_features) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * out_features) {
        size_t col = i % out_features;
        out[i] = __hadd(out[i], bias[col]);
    }
}

template <>
__global__ void bias_add_kernel<__nv_bfloat16>(__nv_bfloat16 *out, const __nv_bfloat16 *bias, size_t batch, size_t out_features) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * out_features) {
        size_t col = i % out_features;
        out[i] = __hadd(out[i], bias[col]);
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t batch, size_t in_features, size_t out_features, int device_id) {
    // Get cuBLAS handle from resource
    auto &res = llaisys::device::nvidia::getResource(device_id);
    cublasHandle_t handle = res.cublasHandle();

    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;

    cublasStatus_t status;

    // Perform matrix multiplication: out = in * weight^T
    // cuBLAS uses column-major, so we compute: out^T = weight * in^T
    switch (type) {
    case LLAISYS_DTYPE_F32:
        status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            out_features, batch, in_features,
                            &alpha_f,
                            (const float *)weight, in_features,
                            (const float *)in, in_features,
                            &beta_f,
                            (float *)out, out_features);
        break;
    case LLAISYS_DTYPE_F64:
        status = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            out_features, batch, in_features,
                            &alpha_d,
                            (const double *)weight, in_features,
                            (const double *)in, in_features,
                            &beta_d,
                            (double *)out, out_features);
        break;
    case LLAISYS_DTYPE_F16:
        status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             out_features, batch, in_features,
                             &alpha_f,
                             weight, CUDA_R_16F, in_features,
                             in, CUDA_R_16F, in_features,
                             &beta_f,
                             out, CUDA_R_16F, out_features,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        break;
    case LLAISYS_DTYPE_BF16:
        // Use cublasGemmEx for BF16
        status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             out_features, batch, in_features,
                             &alpha_f,
                             weight, CUDA_R_16BF, in_features,
                             in, CUDA_R_16BF, in_features,
                             &beta_f,
                             out, CUDA_R_16BF, out_features,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        break;
    default:
        std::cerr << "[ERROR] Unsupported data type for linear: " << type << std::endl;
        throw std::runtime_error("Unsupported data type");
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[ERROR] cuBLAS gemm failed: " << status << std::endl;
        throw std::runtime_error("cuBLAS gemm failed");
    }

    // Add bias if provided
    if (bias != nullptr) {
        size_t total = batch * out_features;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;

        switch (type) {
        case LLAISYS_DTYPE_F32:
            bias_add_kernel<float><<<gridSize, blockSize>>>((float *)out, (const float *)bias, batch, out_features);
            break;
        case LLAISYS_DTYPE_F64:
            bias_add_kernel<double><<<gridSize, blockSize>>>((double *)out, (const double *)bias, batch, out_features);
            break;
        case LLAISYS_DTYPE_F16:
            bias_add_kernel<__half><<<gridSize, blockSize>>>((__half *)out, (const __half *)bias, batch, out_features);
            break;
        case LLAISYS_DTYPE_BF16:
            bias_add_kernel<__nv_bfloat16><<<gridSize, blockSize>>>((__nv_bfloat16 *)out, (const __nv_bfloat16 *)bias, batch, out_features);
            break;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

} // namespace llaisys::ops::nvidia
