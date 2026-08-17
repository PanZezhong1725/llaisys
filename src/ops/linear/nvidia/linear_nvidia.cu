#include "linear_nvidia.hpp"

#include "../../../utils.hpp"
#include "../../../utils/cuda_check.hpp"
#include "../../../utils/cublas_utils.hpp"

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// 加 bias 的逐元素 kernel,（每列共享同一个偏置）
template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    out[idx] = static_cast<T>(static_cast<float>(out[idx]) + static_cast<float>(bias[idx % N]));
}

namespace llaisys::ops::nvidia {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias,
            llaisysDataType_t type, size_t numel) {
    // 形状：in[M,K], weight[N,K], out[M,N]
    const size_t M = in->shape()[0];
    const size_t K = in->shape()[1];
    const size_t N = weight->shape()[0];

    // 获取缓存的 cuBLAS 句柄（进程内只创建一次）
    cublasHandle_t handle = get_cublas_handle();

    // 映射到 cuBLAS 的数据类型
    cudaDataType_t ctype;
    switch (type) {
    case LLAISYS_DTYPE_F32:  ctype = CUDA_R_32F; break;
    case LLAISYS_DTYPE_F16:  ctype = CUDA_R_16F; break;
    case LLAISYS_DTYPE_BF16: ctype = CUDA_R_16BF; break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    // cublasGemmEx 计算 out = in @ weight^T
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,                  // weight 转置，in 不转置
        static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),  // m,n,k
        &alpha,                                    // alpha（void* 指向 float）
        reinterpret_cast<const void *>(weight->data()), ctype, static_cast<int>(K),  // A=weight, lda=K
        reinterpret_cast<const void *>(in->data()),     ctype, static_cast<int>(K),  // B=in,     ldb=K
        &beta,                                     // beta
        reinterpret_cast<void *>(out->data()),          ctype, static_cast<int>(N),  // C=out,    ldc=N
        CUBLAS_COMPUTE_32F,                        // 累加用 fp32
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));           // 走 Tensor Core（f16/bf16 快）

    // 加 bias
    if (bias) {
        const int threads = 256;
        const size_t total = M * N;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        switch (type) {
        case LLAISYS_DTYPE_F32:
            add_bias_kernel<float><<<blocks, threads>>>(
                reinterpret_cast<float *>(out->data()),
                reinterpret_cast<const float *>(bias->data()), M, N);
            break;
        case LLAISYS_DTYPE_F16:
            add_bias_kernel<__half><<<blocks, threads>>>(
                reinterpret_cast<__half *>(out->data()),
                reinterpret_cast<const __half *>(bias->data()), M, N);
            break;
        case LLAISYS_DTYPE_BF16:
            add_bias_kernel<__nv_bfloat16><<<blocks, threads>>>(
                reinterpret_cast<__nv_bfloat16 *>(out->data()),
                reinterpret_cast<const __nv_bfloat16 *>(bias->data()), M, N);
            break;
        default:
            break;
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // 等待gpu, 句柄缓存复用, 不销毁
    CHECK_CUDA(cudaDeviceSynchronize());
}

} // namespace llaisys::ops::nvidia
