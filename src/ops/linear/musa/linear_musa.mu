#include "linear_musa.hpp"

#include "../../../utils.hpp"
#include "../../../utils/musa_check.hpp"
#include "../../../utils/mublas_utils.hpp"   // CHECK_MUBLAS + get_mublas_handle

#include <cstdint>

#include <musa_fp16.h>
#include <musa_bf16.h>

// 加 bias 的逐元素 kernel（每列共享同一个偏置）
template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    out[idx] = static_cast<T>(static_cast<float>(out[idx]) + static_cast<float>(bias[idx % N]));
}

namespace llaisys::ops::musa {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias,
            llaisysDataType_t type, size_t numel) {
    // 形状：in[M,K], weight[N,K], out[M,N]（与 nvidia 版完全相同）
    const size_t M = in->shape()[0];
    const size_t K = in->shape()[1];
    const size_t N = weight->shape()[0];

    // 获取缓存的 mublas 句柄（进程内只创建一次）
    mublasHandle_t handle = get_mublas_handle();

    // 映射到 mublas 的数据类型
    musaDataType_t ctype;
    switch (type) {
    case LLAISYS_DTYPE_F32:  ctype = MUSA_R_32F;  break;
    case LLAISYS_DTYPE_F16:  ctype = MUSA_R_16F;  break;
    case LLAISYS_DTYPE_BF16: ctype = MUSA_R_16BF; break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    mublasComputeType_t compute;
    switch (type) {
    case LLAISYS_DTYPE_F32:  compute = MUBLAS_COMPUTE_32F; break;
    case LLAISYS_DTYPE_F16:  compute = MUBLAS_COMPUTE_16F; break;
    case LLAISYS_DTYPE_BF16: compute = MUBLAS_COMPUTE_32F; break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    // mublas 未实现 f16 的 Tensor Core 路径（f32 的 TENSOR_OP 可用）→ f16 用默认 algo
    const mublasGemmAlgo_t algo = (type == LLAISYS_DTYPE_F16)
        ? MUBLAS_GEMM_DEFAULT
        : MUBLAS_GEMM_DEFAULT_TENSOR_OP;

    const float alpha = 1.0f, beta = 0.0f;
    CHECK_MUBLAS(mublasGemmEx(
        handle,
        MUBLAS_OP_T, MUBLAS_OP_N,                  // weight 转置，in 不转置
        static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),  // m,n,k
        &alpha,
        reinterpret_cast<const void *>(weight->data()), ctype, static_cast<int>(K),  // A=weight, lda=K
        reinterpret_cast<const void *>(in->data()),     ctype, static_cast<int>(K),  // B=in,     ldb=K
        &beta,
        reinterpret_cast<void *>(out->data()),          ctype, static_cast<int>(N),  // C=out,    ldc=N
        compute,                                   // 累加精度：f16→16F，bf16/f32→32F
        algo));                                    // f16→DEFAULT，f32/bf16→TENSOR_OP

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
            add_bias_kernel<__mt_bfloat16><<<blocks, threads>>>(
                reinterpret_cast<__mt_bfloat16 *>(out->data()),
                reinterpret_cast<const __mt_bfloat16 *>(bias->data()), M, N);
            break;
        default:
            break;
        }
        CHECK_MUSA(musaGetLastError());
        CHECK_MUSA(musaDeviceSynchronize());
    }

    // 等待 GPU 算完（句柄缓存复用，不销毁）
    CHECK_MUSA(musaDeviceSynchronize());
}

} // namespace llaisys::ops::musa
