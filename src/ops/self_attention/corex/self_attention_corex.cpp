#include "self_attention_corex.cuh"

#include "../../corex_common.cuh"

#include <cfloat>
#include <cmath>

namespace llaisys::ops::corex {
namespace {

template <typename T>
__global__ void calculateScores(float *scores, const T *query, const T *key,
                                float scale, size_t query_length,
                                size_t key_length, size_t head_count,
                                size_t kv_head_count, size_t head_width) {
    const size_t row = blockIdx.x;
    const size_t query_position = row / head_count;
    const size_t query_head = row % head_count;
    if (query_position >= query_length) {
        return;
    }
    const size_t kv_head = query_head / (head_count / kv_head_count);
    const size_t visible_keys = key_length - query_length + query_position + 1;
    for (size_t key_position = threadIdx.x; key_position < visible_keys;
         key_position += blockDim.x) {
        float dot_product = 0.0f;
        for (size_t d = 0; d < head_width; ++d) {
            const size_t query_index =
                (query_position * head_count + query_head) * head_width + d;
            const size_t key_index =
                (key_position * kv_head_count + kv_head) * head_width + d;
            dot_product +=
                toFloat(query[query_index]) * toFloat(key[key_index]);
        }
        const T rounded_dot = fromFloat<T>(dot_product);
        const T rounded_score = fromFloat<T>(toFloat(rounded_dot) * scale);
        scores[row * key_length + key_position] = toFloat(rounded_score);
    }
}

template <typename T>
__global__ void normalizeScores(float *scores, size_t query_length,
                                size_t key_length, size_t head_count) {
    __shared__ float partial[BLOCK_SIZE];
    const size_t row = blockIdx.x;
    const size_t query_position = row / head_count;
    if (query_position >= query_length) {
        return;
    }
    const size_t visible_keys = key_length - query_length + query_position + 1;
    float local_maximum = -FLT_MAX;
    for (size_t key_position = threadIdx.x; key_position < visible_keys;
         key_position += blockDim.x) {
        local_maximum =
            fmaxf(local_maximum, scores[row * key_length + key_position]);
    }
    partial[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = partial[0];
    float local_sum = 0.0f;
    for (size_t key_position = threadIdx.x; key_position < visible_keys;
         key_position += blockDim.x) {
        const size_t i = row * key_length + key_position;
        const float exponent = expf(scores[i] - maximum);
        scores[i] = exponent;
        local_sum += exponent;
    }
    partial[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inverse_sum = 1.0f / partial[0];
    for (size_t key_position = threadIdx.x; key_position < visible_keys;
         key_position += blockDim.x) {
        const size_t i = row * key_length + key_position;
        scores[i] = toFloat(fromFloat<T>(scores[i] * inverse_sum));
    }
}

template <typename T>
__global__ void combineValues(T *out, const float *scores, const T *value,
                              size_t query_length, size_t key_length,
                              size_t head_count, size_t kv_head_count,
                              size_t value_width) {
    const size_t row = blockIdx.x;
    const size_t query_position = row / head_count;
    const size_t query_head = row % head_count;
    if (query_position >= query_length) {
        return;
    }
    const size_t kv_head = query_head / (head_count / kv_head_count);
    const size_t visible_keys = key_length - query_length + query_position + 1;
    for (size_t column = threadIdx.x; column < value_width;
         column += blockDim.x) {
        float result = 0.0f;
        for (size_t key_position = 0; key_position < visible_keys;
             ++key_position) {
            const size_t value_index =
                (key_position * kv_head_count + kv_head) * value_width + column;
            result += scores[row * key_length + key_position]
                      * toFloat(value[value_index]);
        }
        out[row * value_width + column] = fromFloat<T>(result);
    }
}

template <typename T>
void dispatchAttention(std::byte *out, const std::byte *q,
                       const std::byte *k, const std::byte *v, float scale,
                       size_t query_len, size_t kv_len, size_t num_heads,
                       size_t num_kv_heads, size_t head_dim,
                       size_t value_dim) {
    const size_t row_count = query_len * num_heads;
    float *scores = nullptr;
    COREX_CHECK(cudaMalloc(&scores, row_count * kv_len * sizeof(float)));
    const cudaStream_t stream = currentStream();
    calculateScores<<<static_cast<unsigned int>(row_count), BLOCK_SIZE, 0,
                      stream>>>(
        scores, reinterpret_cast<const T *>(q), reinterpret_cast<const T *>(k),
        scale, query_len, kv_len, num_heads, num_kv_heads, head_dim);
    checkKernel();
    normalizeScores<T><<<static_cast<unsigned int>(row_count), BLOCK_SIZE, 0,
                         stream>>>(scores, query_len, kv_len, num_heads);
    checkKernel();
    combineValues<<<static_cast<unsigned int>(row_count), BLOCK_SIZE, 0,
                    stream>>>(
        reinterpret_cast<T *>(out), scores, reinterpret_cast<const T *>(v),
        query_len, kv_len, num_heads, num_kv_heads, value_dim);
    checkKernel();
    COREX_CHECK(cudaFree(scores));
}

} // namespace

void self_attention(std::byte *out, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type, float scale,
                    size_t query_len, size_t kv_len, size_t num_heads,
                    size_t num_kv_heads, size_t head_dim, size_t value_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return dispatchAttention<float>(out, q, k, v, scale, query_len,
                                        kv_len, num_heads, num_kv_heads,
                                        head_dim, value_dim);
    case LLAISYS_DTYPE_F16:
        return dispatchAttention<__half>(out, q, k, v, scale, query_len,
                                         kv_len, num_heads, num_kv_heads,
                                         head_dim, value_dim);
    case LLAISYS_DTYPE_BF16:
        return dispatchAttention<__nv_bfloat16>(
            out, q, k, v, scale, query_len, kv_len, num_heads,
            num_kv_heads, head_dim, value_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::corex
