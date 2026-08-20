#include "linear_iluvatar.cuh"

#include "../../../utils.hpp"
#include "../../../device/iluvatar/iluvatar_utils.cuh"
#include "../../../device/iluvatar/iluvatar_resource.cuh"
#include "../../../device/iluvatar/iluvatar_dtype.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace llaisys::ops::iluvatar {

namespace {

using llaisys::device::iluvatar::to_float;
using llaisys::device::iluvatar::from_float;

/*
 *  cublasGemmEx() 允许分别指定 A/B/C dtype 以及 compute type；
 *  官方表格明确支持 FP16、BF16、FP32 输入输出配 CUBLAS_COMPUTE_32F
 */
cudaDataType_t toCudaDataType(
    llaisysDataType_t dtype
) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}


void gemm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t m_,
    size_t n_,
    size_t k_,
    cudaStream_t stream
) {
    const int m = static_cast<int>(m_);
    const int n = static_cast<int>(n_);
    const int k = static_cast<int>(k_);

    // 获取当前线程/当前设备的缓存句柄
    cublasHandle_t handle = llaisys::device::iluvatar::getCublasHandle();
    // 现场绑定当前流
    ILUVATAR_CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    const cudaDataType_t data_type = toCudaDataType(dtype);

    /*
     * LLAISYS 行主序: out = in @ weight^T
     * cuBLAS 列主序: out^T = weight @ in^T

      原始：
        X = [m,k]
        W = [n,k]
        Y = X W^T = [m,n]
      转置：
        Y^T = W X^T
        [n,m] =[n,k] × [k,m]

        但是 weight 内存被 cuBLAS 看成：[k,n] = W^T，就需要 CUBLAS_OP_T 转置
        in 内存 cuBLAS 本身看成： [k,m] = X^T，不需要转置
        Y 内存在 cuBLAS 中得到 [n, m] 但是列主序，再行主序读取后就是[m, n]
     */
    ILUVATAR_CUBLAS_CHECK(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            weight,
            data_type,
            k,
            in,
            data_type,
            k,
            &beta,
            out,
            data_type,
            n,
#if !defined(CUBLAS_VERSION) || CUBLAS_VERSION < 11000
            CUDA_R_32F,
#else
            CUBLAS_COMPUTE_32F,
#endif
            CUBLAS_GEMM_DEFAULT
        )
    );
}


template <typename T>
__global__ void add_bias_kernel(
    T *out,
    const T *bias,
    size_t numel,
    size_t n
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= numel) {
        return;
    }

    const size_t col = idx % n;

    const float y = to_float<T>(out[idx]);

    const float b = to_float<T>(bias[col]);

    out[idx] = from_float<T>(y + b);
}

} // namespace


void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t dtype,
    size_t m,
    size_t n,
    size_t k,
    llaisysStream_t stream
){
    cudaStream_t cuda_stream =reinterpret_cast<cudaStream_t>(stream);

    gemm(
        out,
        in,
        weight,
        dtype,
        m,
        n,
        k,
        cuda_stream
    );

    if (bias == nullptr) {
        return;
    }

    const size_t numel = m * n;

    constexpr int BLOCK_SIZE = 256;

    const int grid = static_cast<int>((numel + BLOCK_SIZE - 1)/ BLOCK_SIZE);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        add_bias_kernel<float>
            <<<grid,
               BLOCK_SIZE,
               0,
               cuda_stream>>>(
                reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(bias),
                numel,
                n
            );
        break;
    case LLAISYS_DTYPE_F16:
        add_bias_kernel<__half>
            <<<grid,
               BLOCK_SIZE,
               0,
               cuda_stream>>>(
                reinterpret_cast<__half *>(out),
                reinterpret_cast<const __half *>(bias),
                numel,
                n
            );
        break;
    case LLAISYS_DTYPE_BF16:
        add_bias_kernel<__nv_bfloat16>
            <<<grid,
               BLOCK_SIZE,
               0,
               cuda_stream>>>(
                reinterpret_cast<__nv_bfloat16 *>(out),
                reinterpret_cast<const __nv_bfloat16 *>(bias),
                numel,
                n
            );
        break;
    default:
        throw std::runtime_error(
            "Unsupported dtype for Iluvatar linear"
        );
    }

    ILUVATAR_CUDA_KERNEL_CHECK();
}

}
