#include "self_attention_nvidia.hpp"

#include "../../../device/nvidia/nvidia_resource.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <cstdint>
#include <type_traits>

namespace llaisys::ops::nvidia {

// Apply causal mask + softmax to a scores matrix [seq_len, kv_len] (row-major, type T).
// One block per row.
template <typename T>
__global__ void softmax_causal_kernel(T *scores, size_t seq_len, size_t kv_len) {
    size_t i = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t diag_shift = kv_len - seq_len;
    size_t max_j = i + diag_shift;

    T *row = scores + i * kv_len;

    // Apply causal mask: position i can only attend to j <= i + (kv_len - seq_len)
    for (size_t j = tid; j < kv_len; j += blockDim.x) {
        if (j > max_j) {
            row[j] = static_cast<T>(-INFINITY);
        }
    }
    __syncthreads();

    // Find row max for numerical stability.
    float local_max = -INFINITY;
    for (size_t j = tid; j < kv_len; j += blockDim.x) {
        local_max = fmaxf(local_max, static_cast<float>(row[j]));
    }
    __shared__ float s_max[256];
    s_max[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = s_max[0];
    __syncthreads();

    // Compute exp and sum.
    float local_sum = 0.0f;
    for (size_t j = tid; j < kv_len; j += blockDim.x) {
        float v = __expf(static_cast<float>(row[j]) - row_max);
        row[j] = static_cast<T>(v);
        local_sum += v;
    }
    __shared__ float s_sum[256];
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = s_sum[0];
    __syncthreads();

    // Normalize.
    for (size_t j = tid; j < kv_len; j += blockDim.x) {
        row[j] = static_cast<T>(static_cast<float>(row[j]) / row_sum);
    }
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
static void launch_self_attention(std::byte *out, const std::byte *q, const std::byte *k,
                                  const std::byte *v, size_t seq_len, size_t kv_len,
                                  size_t num_heads, size_t num_kv_heads, size_t head_dim,
                                  float scale) {
    cublasHandle_t handle = llaisys::device::nvidia::getCublasHandle();

    size_t num_repeats = num_heads / num_kv_heads;

    // Temporary scores buffer [seq_len, kv_len] in the same dtype as the inputs.
    T *scores = nullptr;
    cudaMalloc(&scores, seq_len * kv_len * sizeof(T));

    cudaDataType_t type = cuda_type<T>();
    float alpha = 0.0f;
    float beta = 0.0f;

    // For FP32, use the default algorithm to avoid TF32 (reduced precision).
    // For FP16/BF16, use tensor cores for performance.
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    if constexpr (std::is_same_v<T, float>) {
        algo = CUBLAS_GEMM_DEFAULT;
    }

    for (size_t h = 0; h < num_heads; h++) {

        size_t src_kv_head = h / num_repeats;

        // Tensors are stored as [seq_len, num_heads, head_dim].
        // For head h, the data for all sequence positions are at offset h*head_dim,
        // with stride num_heads*head_dim between consecutive sequence positions.
        const T *q_h = reinterpret_cast<const T *>(q) + h * head_dim;
        const T *k_h = reinterpret_cast<const T *>(k) + src_kv_head * head_dim;
        const T *v_h = reinterpret_cast<const T *>(v) + src_kv_head * head_dim;
        T *out_h = reinterpret_cast<T *>(out) + h * head_dim;

        // Leading dimension (stride between rows in the 2D matrix view):
        // Q, out: [seq_len, num_heads, head_dim] → stride = num_heads * head_dim
        // K, V:   [kv_len, num_kv_heads, head_dim] → stride = num_kv_heads * head_dim
        size_t ld_qo = num_heads * head_dim;
        size_t ld_kv = num_kv_heads * head_dim;

        // GEMM1: scores = Q_h @ K_h^T * scale
        // Using column-major cuBLAS: scores^T = K_h @ Q_h^T
        alpha = scale;
        beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     static_cast<int>(kv_len), static_cast<int>(seq_len), static_cast<int>(head_dim),
                     &alpha,
                     k_h, type, static_cast<int>(ld_kv),
                     q_h, type, static_cast<int>(ld_qo),
                     &beta,
                     scores, type, static_cast<int>(kv_len),
                     CUBLAS_COMPUTE_32F, algo);

        // Causal mask + softmax over the kv dimension.
        softmax_causal_kernel<T><<<static_cast<int>(seq_len), 256>>>(scores, seq_len, kv_len);

        // GEMM2: out_h = scores @ V_h
        alpha = 1.0f;
        beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     static_cast<int>(head_dim), static_cast<int>(seq_len), static_cast<int>(kv_len),
                     &alpha,
                     v_h, type, static_cast<int>(ld_kv),
                     scores, type, static_cast<int>(kv_len),
                     &beta,
                     out_h, type, static_cast<int>(ld_qo),
                     CUBLAS_COMPUTE_32F, algo);
    }

    cudaFree(scores);
}

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t seq_len, size_t kv_len, size_t num_heads,
                    size_t num_kv_heads, size_t head_dim, float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_self_attention<float>(out, q, k, v, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_F16:
        return launch_self_attention<__half>(out, q, k, v, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return launch_self_attention<__nv_bfloat16>(out, q, k, v, seq_len, kv_len, num_heads, num_kv_heads, head_dim, scale);
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
