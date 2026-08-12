#pragma once

#include <cublas_v2.h>

#include <iostream>
#include <stdexcept>

// cuBLAS 错误检查宏：调用 cuBLAS 函数，若失败则打印错误码并抛异常。
// 与 cuda_check.hpp 的 CHECK_CUDA 对应，但针对 cuBLAS 的状态码（cublasStatus_t）。
#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        cublasStatus_t st___ = (call);                                           \
        if (st___ != CUBLAS_STATUS_SUCCESS) {                                    \
            std::cerr << "[ERROR] cuBLAS error code: " << (int)st___             \
                      << " at " << __FILE__ << ":" << __LINE__ << "." << std::endl; \
            throw std::runtime_error("cuBLAS error");                            \
        }                                                                        \
    } while (0)

// 进程内只创建一次的 cuBLAS 句柄（Meyers singleton）。
// linear / self_attention 等算子共用，避免每次调用 create/destroy 的开销。
inline cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS create failed");
        }
        return h;
    }();
    return handle;
}
