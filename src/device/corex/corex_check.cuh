#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define COREX_CHECK(call)                                                       \
    do {                                                                        \
        const cudaError_t status_ = (call);                                     \
        if (status_ != cudaSuccess) {                                           \
            throw std::runtime_error(                                           \
                std::string("CoreX runtime error at ") + __FILE__ + ":"       \
                + std::to_string(__LINE__) + ": "                             \
                + cudaGetErrorString(status_));                                \
        }                                                                       \
    } while (false)

#define COREX_BLAS_CHECK(call)                                                  \
    do {                                                                        \
        const cublasStatus_t status_ = (call);                                  \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error(                                           \
                std::string("CoreX cuBLAS error at ") + __FILE__ + ":"        \
                + std::to_string(__LINE__) + ": status "                       \
                + std::to_string(static_cast<int>(status_)));                   \
        }                                                                       \
    } while (false)
