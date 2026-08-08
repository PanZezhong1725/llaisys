// src/ops/self_attention/nvidia/self_attention_nvidia.cu
#include "self_attention_nvidia.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cmath>

namespace llaisys::ops::nvidia {

// Causal softmax kernel
template <typename T>
__global__ void causal_softmax_kernel(T *attn_scores, size_t qlen, size_t kvlen, size_t nh) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < qlen * nh) {
        size_t q_idx = i / nh;
        size_t h_idx = i % nh;
        size_t key_limit = q_idx + (kvlen - qlen);

        // Find max for numerical stability
        float max_val = -INFINITY;
        for (size_t j = 0; j <= key_limit && j < kvlen; j++) {
            size_t idx = q_idx * kvlen * nh + j * nh + h_idx;
            float val = static_cast<float>(attn_scores[idx]);
            max_val = fmaxf(max_val, val);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (size_t j = 0; j <= key_limit && j < kvlen; j++) {
            size_t idx = q_idx * kvlen * nh + j * nh + h_idx;
            float val = static_cast<float>(attn_scores[idx]);
            float exp_val = expf(val - max_val);
            attn_scores[idx] = static_cast<T>(exp_val);
            sum += exp_val;
        }

        // Normalize
        for (size_t j = 0; j <= key_limit && j < kvlen; j++) {
            size_t idx = q_idx * kvlen * nh + j * nh + h_idx;
            float val = static_cast<float>(attn_scores[idx]);
            attn_scores[idx] = static_cast<T>(val / sum);
        }
    }
}

// Q * K^T kernel
template <typename T>
__global__ void qk_matmul_kernel(const T *q, const T *k, T *scores, float scale, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < qlen * kvlen * nh) {
        size_t q_idx = i / (kvlen * nh);
        size_t k_idx = (i / nh) % kvlen;
        size_t h_idx = i % nh;
        size_t kvh_idx = h_idx / (nh / nkvh);
        
        float sum = 0.0f;
        for (size_t d = 0; d < hd; d++) {
            float q_val = static_cast<float>(q[q_idx * nh * hd + h_idx * hd + d]);
            float k_val = static_cast<float>(k[k_idx * nkvh * hd + kvh_idx * hd + d]);
            sum += q_val * k_val;
        }
        
        scores[i] = static_cast<T>(sum * scale);
    }
}

// Scores * V kernel
template <typename T>
__global__ void sv_matmul_kernel(const T *scores, const T *v, T *out, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < qlen * nh * hd) {
        size_t q_idx = i / (nh * hd);
        size_t h_idx = (i / hd) % nh;
        size_t d_idx = i % hd;
        size_t kvh_idx = h_idx / (nh / nkvh);
        size_t key_limit = q_idx + (kvlen - qlen);

        float sum = 0.0f;
        for (size_t j = 0; j <= key_limit && j < kvlen; j++) {
            float s_val = static_cast<float>(scores[q_idx * kvlen * nh + j * nh + h_idx]);
            float v_val = static_cast<float>(v[j * nkvh * hd + kvh_idx * hd + d_idx]);
            sum += s_val * v_val;
        }
        
        out[i] = static_cast<T>(sum);
    }
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    // Allocate temporary storage for attention scores
    void *scores;
    size_t scores_size = qlen * kvlen * nh;
    size_t element_size;
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        element_size = sizeof(float);
        break;
    case LLAISYS_DTYPE_F16:
        element_size = sizeof(__half);
        break;
    case LLAISYS_DTYPE_BF16:
        element_size = sizeof(__nv_bfloat16);
        break;
    case LLAISYS_DTYPE_F64:
        element_size = sizeof(double);
        break;
    default:
        std::cerr << "[ERROR] Unsupported data type for self_attention: " << type << std::endl;
        throw std::runtime_error("Unsupported data type");
    }
    
    cudaMalloc(&scores, scores_size * element_size);
    
    int blockSize = 256;
    
    // Step 1: Compute Q * K^T
    {
        size_t total = qlen * kvlen * nh;
        int gridSize = (total + blockSize - 1) / blockSize;
        
        switch (type) {
        case LLAISYS_DTYPE_F32:
            qk_matmul_kernel<float><<<gridSize, blockSize>>>(
                (const float *)q, (const float *)k, (float *)scores, scale, qlen, kvlen, nh, nkvh, hd);
            break;
        case LLAISYS_DTYPE_F16:
            qk_matmul_kernel<__half><<<gridSize, blockSize>>>(
                (const __half *)q, (const __half *)k, (__half *)scores, scale, qlen, kvlen, nh, nkvh, hd);
            break;
        case LLAISYS_DTYPE_BF16:
            qk_matmul_kernel<__nv_bfloat16><<<gridSize, blockSize>>>(
                (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k, (__nv_bfloat16 *)scores, scale, qlen, kvlen, nh, nkvh, hd);
            break;
        case LLAISYS_DTYPE_F64:
            qk_matmul_kernel<double><<<gridSize, blockSize>>>(
                (const double *)q, (const double *)k, (double *)scores, scale, qlen, kvlen, nh, nkvh, hd);
            break;
        }
    }
    
    // Step 2: Apply causal softmax
    {
        size_t total = qlen * nh;
        int gridSize = (total + blockSize - 1) / blockSize;
        
        switch (type) {
        case LLAISYS_DTYPE_F32:
            causal_softmax_kernel<float><<<gridSize, blockSize>>>((float *)scores, qlen, kvlen, nh);
            break;
        case LLAISYS_DTYPE_F16:
            causal_softmax_kernel<__half><<<gridSize, blockSize>>>((__half *)scores, qlen, kvlen, nh);
            break;
        case LLAISYS_DTYPE_BF16:
            causal_softmax_kernel<__nv_bfloat16><<<gridSize, blockSize>>>((__nv_bfloat16 *)scores, qlen, kvlen, nh);
            break;
        case LLAISYS_DTYPE_F64:
            causal_softmax_kernel<double><<<gridSize, blockSize>>>((double *)scores, qlen, kvlen, nh);
            break;
        }
    }
    
    // Step 3: Compute Scores * V
    {
        size_t total = qlen * nh * hd;
        int gridSize = (total + blockSize - 1) / blockSize;
        
        switch (type) {
        case LLAISYS_DTYPE_F32:
            sv_matmul_kernel<float><<<gridSize, blockSize>>>(
                (const float *)scores, (const float *)v, (float *)attn_val, qlen, kvlen, nh, nkvh, hd);
            break;
        case LLAISYS_DTYPE_F16:
            sv_matmul_kernel<__half><<<gridSize, blockSize>>>(
                (const __half *)scores, (const __half *)v, (__half *)attn_val, qlen, kvlen, nh, nkvh, hd);
            break;
        case LLAISYS_DTYPE_BF16:
            sv_matmul_kernel<__nv_bfloat16><<<gridSize, blockSize>>>(
                (const __nv_bfloat16 *)scores, (const __nv_bfloat16 *)v, (__nv_bfloat16 *)attn_val, qlen, kvlen, nh, nkvh, hd);
            break;
        case LLAISYS_DTYPE_F64:
            sv_matmul_kernel<double><<<gridSize, blockSize>>>(
                (const double *)scores, (const double *)v, (double *)attn_val, qlen, kvlen, nh, nkvh, hd);
            break;
        }
    }
    
    // Free temporary storage
    cudaFree(scores);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

} // namespace llaisys::ops::nvidia
