#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

inline void checkCudaError(
    cudaError_t error,
    const char *expr,
    const char *file,
    int line
) {
    if (error == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string("[CUDA] ")
        + cudaGetErrorName(error)
        + ": "
        + cudaGetErrorString(error)
        + "\n  expression: "
        + expr
        + "\n  location: "
        + file
        + ":"
        + std::to_string(line)
    );
}

} // namespace llaisys::device::nvidia


#define CUDA_CHECK(expr)                                      \
    do {                                                      \
        ::llaisys::device::nvidia::checkCudaError(            \
            (expr),                                           \
            #expr,                                            \
            __FILE__,                                         \
            __LINE__                                          \
        );                                                    \
    } while (0)


#define CUDA_KERNEL_CHECK() \
    CUDA_CHECK(cudaGetLastError())


inline const char *cublasStatusString(
    cublasStatus_t status
) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    default:
        return "UNKNOWN_CUBLAS_STATUS";
    }
}


inline void checkCublasError(
    cublasStatus_t status,
    const char *expr,
    const char *file,
    int line
) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }

    throw std::runtime_error(
        std::string("[cuBLAS] ")
        + cublasStatusString(status)
        + "\n  expression: "
        + expr
        + "\n  location: "
        + file
        + ":"
        + std::to_string(line)
    );
}


#define CUBLAS_CHECK(expr)                   \
    do {                                     \
        checkCublasError(                    \
            (expr),                          \
            #expr,                           \
            __FILE__,                        \
            __LINE__                         \
        );                                   \
    } while (0)
 