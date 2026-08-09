#pragma once

#include "../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <sstream>

namespace llaisys::device::nvidia {

inline void checkCuda(cudaError_t status, const char *expression,
                      const char *file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << "CUDA call failed: " << expression << ": "
            << cudaGetErrorString(status) << " at " << file << ':' << line;
    throw std::runtime_error(message.str());
}

inline void checkCublas(cublasStatus_t status, const char *expression,
                        const char *file, int line) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    std::ostringstream message;
    message << "cuBLAS call failed: " << expression << " (status "
            << static_cast<int>(status) << ") at " << file << ':' << line;
    throw std::runtime_error(message.str());
}

} // namespace llaisys::device::nvidia

#define CUDA_CHECK(expr) \
    ::llaisys::device::nvidia::checkCuda((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) \
    ::llaisys::device::nvidia::checkCublas((expr), #expr, __FILE__, __LINE__)
