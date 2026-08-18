#include "../../../utils.hpp"
#include "linear_cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "../../../device/nvidia/nvidia_resource.cuh"
#define CEIL(a, b) (((a) + (b) - 1) / (b))
// F32/BF16/F16 都走 cublas；原来练手写的 tiled-GEMM kernel 见 git 历史（613f360 之前）。

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < M * N) out[idx] = out[idx] + bias[idx % N];
}


// in : [M, K], weight : [N, K], bias : [N], out : [M, N]；out = in * weight^T + bias
namespace llaisys::ops::cuda {

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K,llaisys::device::DeviceResource *resource) {
    auto handle = static_cast<llaisys::device::nvidia::Resource*>(resource)->cublasHandle();
    switch (type) {
    case LLAISYS_DTYPE_F32:
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        const auto *in_f32 = reinterpret_cast<const float *>(in);
        const auto *weight_f32 = reinterpret_cast<const float *>(weight);
        const auto *bias_f32 = reinterpret_cast<const float *>(bias);
        auto *out_f32 = reinterpret_cast<float *>(out);

        cublasStatus_t status = cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            &alpha,
            weight_f32,
            static_cast<int>(K),
            in_f32,
            static_cast<int>(K),
            &beta,
            out_f32,
            static_cast<int>(N)
        );
        if(bias != nullptr){
        constexpr int block_size = 256;
        int grid_size = static_cast<int>(CEIL(M * N,block_size));
        add_bias_kernel<<<grid_size, block_size>>>(out_f32, bias_f32 , M, N);
        
        }
        return;
    }
    case LLAISYS_DTYPE_BF16:
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        const auto *in_bf16 =  reinterpret_cast<const __nv_bfloat16*>(in);
        const auto *weight_bf16 = reinterpret_cast<const __nv_bfloat16 *>(weight);
        const auto *bias_bf16 = reinterpret_cast<const __nv_bfloat16 *>(bias);
        auto *out_bf16 = reinterpret_cast<__nv_bfloat16 *>(out);

        cublasStatus_t status = cublasSgemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            &alpha,
            weight_bf16,
            CUDA_R_16BF,
            static_cast<int>(K),
            in_bf16,
            CUDA_R_16BF,
            static_cast<int>(K),
            &beta,
            out_bf16,
            CUDA_R_16BF,
            static_cast<int>(N)
        );
        if(bias != nullptr){
        constexpr int block_size = 256;
        int grid_size = static_cast<int>(CEIL(M * N,block_size));

        add_bias_kernel<<<grid_size, block_size>>>(out_bf16, bias_bf16 , M, N);
        
        }
        return;
    }

    case LLAISYS_DTYPE_F16:
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        const auto *in_f16 =  reinterpret_cast<const __half*>(in);
        const auto *weight_f16 = reinterpret_cast<const __half *>(weight);
        const auto *bias_f16 = reinterpret_cast<const __half *>(bias);
        auto *out_f16 = reinterpret_cast<__half *>(out);

        cublasStatus_t status = cublasSgemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            &alpha,
            weight_f16,
            CUDA_R_16F,
            static_cast<int>(K),
            in_f16,
            CUDA_R_16F,
            static_cast<int>(K),
            &beta,
            out_f16,
            CUDA_R_16F,
            static_cast<int>(N)
        );
        if(bias != nullptr){
        constexpr int block_size = 256;
        int grid_size = static_cast<int>(CEIL(M * N,block_size));

        add_bias_kernel<<<grid_size, block_size>>>(out_f16, bias_f16 , M, N);
        
        }
        return;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cuda
