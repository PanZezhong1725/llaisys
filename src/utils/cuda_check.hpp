#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

// 检查 CUDA 调用是否成功，失败则打印错误信息并抛出异常。
// 该宏仅应在 .cu 文件的 host 端代码中使用。
#define CHECK_CUDA(call)                                                            \
    do {                                                                            \
        cudaError_t err___ = (call);                                                \
        if (err___ != cudaSuccess) {                                                \
            std::cerr << "[ERROR] CUDA error: " << cudaGetErrorString(err___)       \
                      << " at " << __FILE__ << ":" << __LINE__ << "." << std::endl; \
            throw std::runtime_error("CUDA runtime error");                         \
        }                                                                           \
    } while (0)
