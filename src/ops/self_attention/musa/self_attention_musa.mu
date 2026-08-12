#include "self_attention_musa.hpp"

#include "../../../utils.hpp"
#include "../../../utils/musa_check.hpp"
#include "../../../utils/mublas_utils.hpp"

#include <cmath>

#include <musa_fp16.h>
#include <musa_bf16.h>

constexpr int THREAD_NUM = 256;

template <typename T>
__global__ void softmax_causal_kernel(T *S, size_t qlen, size_t nh, size_t kvlen) {
    const size_t qpos = blockIdx.x;
    const size_t h    = blockIdx.y;
    const size_t diag = kvlen - qlen;               // 因果偏移
    T *row = S + (qpos * nh + h) * kvlen;

    __shared__ float red[THREAD_NUM];
    __shared__ float row_max;
    __shared__ float inv_sum;

    // 找最大值
    float local_max = -INFINITY;
    for (size_t kpos = threadIdx.x; kpos < kvlen; kpos += THREAD_NUM) {
        float s = static_cast<float>(row[kpos]);
        if (kpos > qpos + diag) s = -INFINITY;
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

namespace llaisys::ops::musa {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v,
                    float scale, llaisysDataType_t type) {
    const size_t qlen  = q->shape()[0];
    const size_t nh    = q->shape()[1];
    const size_t hd    = q->shape()[2];
    const size_t kvlen = k->shape()[0];
    const size_t nkvh  = k->shape()[1];
    const size_t group_size = nh / nkvh;            // GQA
    const size_t es = q->elementSize();

    mublasHandle_t handle = get_mublas_handle();
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

    // 临时缓冲
    tensor_t scores = Tensor::create({qlen, nh, kvlen}, type,
                                     attn_val->deviceType(), attn_val->deviceId());
    void *S = scores->data();

    const float alpha_one = 1.0f, beta_zero = 0.0f;

    // 每个头一次 GEMM
    for (size_t h = 0; h < nh; ++h) {
        const size_t kv_head = h / group_size;
        CHECK_MUBLAS(mublasGemmEx(
            handle,
            MUBLAS_OP_T, MUBLAS_OP_N,
            static_cast<int>(kvlen), static_cast<int>(qlen), static_cast<int>(hd),
            &scale,                                    // alpha = scale
            reinterpret_cast<const char *>(k->data()) + kv_head * hd * es,  ctype, static_cast<int>(nkvh * hd),   // A=K_h
            reinterpret_cast<const char *>(q->data()) + h * hd * es,        ctype, static_cast<int>(nh * hd),    // B=Q_h
            &beta_zero,
            reinterpret_cast<char *>(S) + h * kvlen * es, ctype, static_cast<int>(nh * kvlen),                    // C=S_h
            compute,                                   // 累加精度：f16→16F，bf16/f32→32F
            algo));                                    // f16→DEFAULT，f32/bf16→TENSOR_OP
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
        softmax_causal_kernel<__mt_bfloat16><<<softmax_grid, THREAD_NUM>>>(
            reinterpret_cast<__mt_bfloat16 *>(S), qlen, nh, kvlen);
        break;
    default:
        break;
    }
    CHECK_MUSA(musaGetLastError());

    // 再次逐头 GEMM
    for (size_t h = 0; h < nh; ++h) {
        const size_t kv_head = h / group_size;
        CHECK_MUBLAS(mublasGemmEx(
            handle,
            MUBLAS_OP_N, MUBLAS_OP_N,
            static_cast<int>(hd), static_cast<int>(qlen), static_cast<int>(kvlen),
            &alpha_one,
            reinterpret_cast<const char *>(v->data()) + kv_head * hd * es, ctype, static_cast<int>(nkvh * hd),   // A=V_h
            reinterpret_cast<const char *>(S) + h * kvlen * es,            ctype, static_cast<int>(nh * kvlen),   // B=S_h
            &beta_zero,
            reinterpret_cast<char *>(attn_val->data()) + h * hd * es,       ctype, static_cast<int>(nh * hd),    // C=attn_val_h
            compute,                                   // 累加精度：f16→16F，bf16/f32→32F
            algo));                                    // f16→DEFAULT，f32/bf16→TENSOR_OP
    }

    CHECK_MUSA(musaDeviceSynchronize());
}

} // namespace llaisys::ops::musa
