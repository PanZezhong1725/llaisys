#include "self_attention_nvidia.hpp"

#include "../../../utils.hpp"
#include "../../../utils/cuda_check.hpp"
#include "../../../utils/cublas_utils.hpp"

#include <cmath>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

constexpr int THREAD_NUM = 256;

template <typename T>
__global__ void softmax_causal_kernel(T *S, size_t qlen, size_t nh, size_t kvlen) {
    const size_t qpos = blockIdx.x;
    const size_t h    = blockIdx.y;
    const size_t diag = kvlen - qlen; // 因果偏移
    T *row = S + (qpos * nh + h) * kvlen; // 本 block 负责的分数行

    __shared__ float red[THREAD_NUM]; // 归约缓冲区
    __shared__ float row_max; // 该行最大值
    __shared__ float inv_sum; // 1 / 归一化总和

    // 找最大值
    float local_max = -INFINITY;
    for (size_t kpos = threadIdx.x; kpos < kvlen; kpos += THREAD_NUM) {
        float s = static_cast<float>(row[kpos]);
        if (kpos > qpos + diag) s = -INFINITY;  // 屏蔽掉因果掩码位置
        if (s > local_max) local_max = s;
    }
    red[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = THREAD_NUM / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) row_max = red[0];
    __syncthreads();

    // exp(s - max) 并求和
    float local_sum = 0.0f;
    for (size_t kpos = threadIdx.x; kpos < kvlen; kpos += THREAD_NUM) {
        if (kpos > qpos + diag) { row[kpos] = static_cast<T>(0.0f); continue; }
        float e = __expf(static_cast<float>(row[kpos]) - row_max);
        row[kpos] = static_cast<T>(e);
        local_sum += e;
    }
    red[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = THREAD_NUM / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) inv_sum = 1.0f / red[0];
    __syncthreads();

    // 归一化
    for (size_t kpos = threadIdx.x; kpos < kvlen; kpos += THREAD_NUM) {
        row[kpos] = static_cast<T>(static_cast<float>(row[kpos]) * inv_sum);
    }
}

namespace llaisys::ops::nvidia {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale, llaisysDataType_t type) {
    const size_t qlen  = q->shape()[0];
    const size_t nh    = q->shape()[1];
    const size_t hd    = q->shape()[2];
    const size_t kvlen = k->shape()[0];
    const size_t nkvh  = k->shape()[1];
    const size_t group_size = nh / nkvh; // 几个 q 头共享一个 KV 头
    const size_t es = q->elementSize();  // 元素字节数（f32=4, f16/bf16=2）

    cublasHandle_t handle = get_cublas_handle();
    cudaDataType_t ctype;
    switch (type) {
    case LLAISYS_DTYPE_F32:  ctype = CUDA_R_32F;  break;
    case LLAISYS_DTYPE_F16:  ctype = CUDA_R_16F;  break;
    case LLAISYS_DTYPE_BF16: ctype = CUDA_R_16BF; break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    // 临时缓冲
    tensor_t scores = Tensor::create({qlen, nh, kvlen}, type,
                                     attn_val->deviceType(), attn_val->deviceId());
    void *S = scores->data();

    const float alpha_one = 1.0f, beta_zero = 0.0f;

    // 每个头一次 GEMM
    for (size_t h = 0; h < nh; ++h) {
        const size_t kv_head = h / group_size;
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(kvlen), static_cast<int>(qlen), static_cast<int>(hd),
            &scale,                                    // alpha = scale（缩放因子）
            reinterpret_cast<const char *>(k->data()) + kv_head * hd * es,  ctype, static_cast<int>(nkvh * hd),   // A=K_h, lda
            reinterpret_cast<const char *>(q->data()) + h * hd * es,        ctype, static_cast<int>(nh * hd),    // B=Q_h, ldb
            &beta_zero,
            reinterpret_cast<char *>(S) + h * kvlen * es, ctype, static_cast<int>(nh * kvlen),                    // C=S_h, ldc
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // softmax + 因果掩码
    dim3 softmax_grid(static_cast<unsigned>(qlen), static_cast<unsigned>(nh));
    switch (type) {
    case LLAISYS_DTYPE_F32:
        softmax_causal_kernel<float><<<softmax_grid, THREAD_NUM>>>(
            reinterpret_cast<float *>(S), qlen, nh, kvlen);
        break;
    case LLAISYS_DTYPE_F16:
        softmax_causal_kernel<__half><<<softmax_grid, THREAD_NUM>>>(
            reinterpret_cast<__half *>(S), qlen, nh, kvlen);
        break;
    case LLAISYS_DTYPE_BF16:
        softmax_causal_kernel<__nv_bfloat16><<<softmax_grid, THREAD_NUM>>>(
            reinterpret_cast<__nv_bfloat16 *>(S), qlen, nh, kvlen);
        break;
    default:
        break;
    }
    CHECK_CUDA(cudaGetLastError());

    // 再次逐头 GEMM
    for (size_t h = 0; h < nh; ++h) {
        const size_t kv_head = h / group_size;
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(hd), static_cast<int>(qlen), static_cast<int>(kvlen),
            &alpha_one,
            reinterpret_cast<const char *>(v->data()) + kv_head * hd * es, ctype, static_cast<int>(nkvh * hd),   // A=V_h, lda
            reinterpret_cast<const char *>(S) + h * kvlen * es,            ctype, static_cast<int>(nh * kvlen),   // B=S_h, ldb
            &beta_zero,
            reinterpret_cast<char *>(attn_val->data()) + h * hd * es,       ctype, static_cast<int>(nh * hd),    // C=attn_val_h, ldc
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
}

} // namespace llaisys::ops::nvidia
