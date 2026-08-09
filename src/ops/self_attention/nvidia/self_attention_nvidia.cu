#include "self_attention_nvidia.cuh"

#include "../../nvidia_common.cuh"

#include <cfloat>
#include <cmath>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void attentionScoreKernel(
    float *scores, const T *q, const T *k, float scale, size_t query_len,
    size_t kv_len, size_t num_heads, size_t num_kv_heads,
    size_t head_dim) {
    const size_t row = blockIdx.x;
    const size_t query_pos = row / num_heads;
    const size_t query_head = row % num_heads;
    if (query_pos >= query_len) {
        return;
    }
    const size_t group_size = num_heads / num_kv_heads;
    const size_t kv_head = query_head / group_size;
    const size_t allowed_key_count = kv_len - query_len + query_pos + 1;
    for (size_t key_pos = threadIdx.x; key_pos < allowed_key_count;
         key_pos += blockDim.x) {
        float dot = 0.0f;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            const size_t q_index =
                (query_pos * num_heads + query_head) * head_dim + dim;
            const size_t k_index =
                (key_pos * num_kv_heads + kv_head) * head_dim + dim;
            dot += toFloat(q[q_index]) * toFloat(k[k_index]);
        }
        const T dot_value = fromFloat<T>(dot);
        const T scaled_value = fromFloat<T>(toFloat(dot_value) * scale);
        scores[row * kv_len + key_pos] = toFloat(scaled_value);
    }
}

template <typename T>
__global__ void attentionSoftmaxKernel(
    float *scores, size_t query_len, size_t kv_len, size_t num_heads) {
    __shared__ float reduction[THREADS];
    const size_t row = blockIdx.x;
    const size_t query_pos = row / num_heads;
    if (query_pos >= query_len) {
        return;
    }
    const size_t allowed_key_count = kv_len - query_len + query_pos + 1;
    float local_max = -FLT_MAX;
    for (size_t key_pos = threadIdx.x; key_pos < allowed_key_count;
         key_pos += blockDim.x) {
        local_max = fmaxf(local_max, scores[row * kv_len + key_pos]);
    }
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            reduction[threadIdx.x] =
                fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    const float max_score = reduction[0];
    float local_sum = 0.0f;
    for (size_t key_pos = threadIdx.x; key_pos < allowed_key_count;
         key_pos += blockDim.x) {
        const float value = expf(scores[row * kv_len + key_pos] - max_score);
        scores[row * kv_len + key_pos] = value;
        local_sum += value;
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float inverse_sum = 1.0f / reduction[0];
    for (size_t key_pos = threadIdx.x; key_pos < allowed_key_count;
         key_pos += blockDim.x) {
        const T probability =
            fromFloat<T>(scores[row * kv_len + key_pos] * inverse_sum);
        scores[row * kv_len + key_pos] = toFloat(probability);
    }
}

template <typename T>
__global__ void attentionValueKernel(
    T *out, const float *scores, const T *v, size_t query_len,
    size_t kv_len, size_t num_heads, size_t num_kv_heads,
    size_t value_dim) {
    const size_t row = blockIdx.x;
    const size_t query_pos = row / num_heads;
    const size_t query_head = row % num_heads;
    if (query_pos >= query_len) {
        return;
    }
    const size_t group_size = num_heads / num_kv_heads;
    const size_t kv_head = query_head / group_size;
    const size_t allowed_key_count = kv_len - query_len + query_pos + 1;
    for (size_t column = threadIdx.x; column < value_dim;
         column += blockDim.x) {
        float result = 0.0f;
        for (size_t key_pos = 0; key_pos < allowed_key_count; ++key_pos) {
            const size_t value_index =
                (key_pos * num_kv_heads + kv_head) * value_dim + column;
            result += scores[row * kv_len + key_pos] * toFloat(v[value_index]);
        }
        out[row * value_dim + column] = fromFloat<T>(result);
    }
}

template <typename T>
void launch(std::byte *out, const std::byte *q, const std::byte *k,
            const std::byte *v, float scale, size_t query_len, size_t kv_len,
            size_t num_heads, size_t num_kv_heads, size_t head_dim,
            size_t value_dim) {
    const size_t rows = query_len * num_heads;
    float *scores = nullptr;
    CUDA_CHECK(cudaMalloc(&scores, rows * kv_len * sizeof(float)));
    const auto stream = currentStream();
    attentionScoreKernel<<<static_cast<unsigned int>(rows), THREADS, 0,
                           stream>>>(
        scores, reinterpret_cast<const T *>(q), reinterpret_cast<const T *>(k),
        scale, query_len, kv_len, num_heads, num_kv_heads, head_dim);
    checkKernelLaunch();
    attentionSoftmaxKernel<T><<<static_cast<unsigned int>(rows), THREADS, 0,
                                stream>>>(
        scores, query_len, kv_len, num_heads);
    checkKernelLaunch();
    attentionValueKernel<<<static_cast<unsigned int>(rows), THREADS, 0,
                           stream>>>(
        reinterpret_cast<T *>(out), scores, reinterpret_cast<const T *>(v),
        query_len, kv_len, num_heads, num_kv_heads, value_dim);
    checkKernelLaunch();
    CUDA_CHECK(cudaFree(scores));
}

} // namespace

void self_attention(std::byte *out, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type, float scale,
                    size_t query_len, size_t kv_len, size_t num_heads,
                    size_t num_kv_heads, size_t head_dim, size_t value_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, q, k, v, scale, query_len, kv_len,
                             num_heads, num_kv_heads, head_dim, value_dim);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, q, k, v, scale, query_len, kv_len,
                              num_heads, num_kv_heads, head_dim, value_dim);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(
            out, q, k, v, scale, query_len, kv_len, num_heads,
            num_kv_heads, head_dim, value_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
