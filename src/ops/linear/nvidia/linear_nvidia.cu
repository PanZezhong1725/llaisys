#include "linear_nvidia.hpp"

#include "../../../device/nvidia/nvidia_resource.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <cstdint>
#include <type_traits>

namespace llaisys::ops::nvidia {

// Add bias to the GEMM output: out[i][j] += bias[j]
template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t seq_len, size_t out_features) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * out_features;
    if (idx >= total) {
        return;
    }
    size_t j = idx % out_features;
    out[idx] = static_cast<T>(static_cast<float>(out[idx]) + static_cast<float>(bias[j]));
}

template <typename T>
static cudaDataType_t cuda_type() {
    if constexpr (std::is_same_v<T, float>) {
        return CUDA_R_32F;
    } else if constexpr (std::is_same_v<T, __half>) {
        return CUDA_R_16F;
    } else {
        return CUDA_R_16BF;
    }
}

template <typename T>
static void launch_linear(std::byte *out, const std::byte *in, const std::byte *weight,
                          const std::byte *bias, size_t seq_len, size_t in_features,
                          size_t out_features, bool has_bias) {
    cublasHandle_t handle = llaisys::device::nvidia::getCublasHandle();

    // out = in @ weight^T  (all row-major)
    // Using column-major cuBLAS: out^T = weight @ in^T
    //   cublasGemmEx(T, N, out_features, seq_len, in_features, ...)
    //   A = weight (op=T, lda = in_features), B = in (op=N, ldb = in_features), C = out (ldc = out_features)
    int m = static_cast<int>(out_features);
    int n = static_cast<int>(seq_len);
    int k = static_cast<int>(in_features);

    float alpha = 1.0f;
    float beta = 0.0f;
    cudaDataType_t type = cuda_type<T>();

    // For FP32, use the default algorithm to avoid TF32 (reduced precision).
    // For FP16/BF16, use tensor cores for performance.
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    if constexpr (std::is_same_v<T, float>) {
        algo = CUBLAS_GEMM_DEFAULT;
    }

    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 m, n, k,
                 &alpha,
                 weight, type, k,
                 in, type, k,
                 &beta,
                 out, type, m,
                 CUBLAS_COMPUTE_32F, algo);



    if (has_bias) {
        size_t total = seq_len * out_features;
        const int block = 256;
        int grid = static_cast<int>((total + block - 1) / block);
        add_bias_kernel<T><<<grid, block>>>(reinterpret_cast<T *>(out),
                                            reinterpret_cast<const T *>(bias),
                                            seq_len, out_features);
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t seq_len, size_t in_features, size_t out_features,
            bool has_bias) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_linear<float>(out, in, weight, bias, seq_len, in_features, out_features, has_bias);
    case LLAISYS_DTYPE_F16:
        return launch_linear<__half>(out, in, weight, bias, seq_len, in_features, out_features, has_bias);
    case LLAISYS_DTYPE_BF16:
        return launch_linear<__nv_bfloat16>(out, in, weight, bias, seq_len, in_features, out_features, has_bias);
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
