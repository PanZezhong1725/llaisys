#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(
    T *out,
    const T *q,
    const T *k,
    const T *v,
    float scale,
    size_t query_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t value_dim) {
    const size_t cache_offset = kv_len - query_len;
    const size_t group_size = num_heads / num_kv_heads;
    std::vector<float> scores(kv_len);

    for (size_t query_pos = 0; query_pos < query_len; ++query_pos) {
        const size_t allowed_key_count = cache_offset + query_pos + 1;

        for (size_t query_head = 0; query_head < num_heads; ++query_head) {
            const size_t kv_head = query_head / group_size;
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t key_pos = 0; key_pos < allowed_key_count; ++key_pos) {
                float dot = 0.0f;

                for (size_t dim = 0; dim < head_dim; ++dim) {
                    const size_t q_index =
                        (query_pos * num_heads + query_head) * head_dim + dim;
                    const size_t k_index =
                        (key_pos * num_kv_heads + kv_head) * head_dim + dim;
                    dot += llaisys::utils::cast<float>(q[q_index])
                           * llaisys::utils::cast<float>(k[k_index]);
                }

                scores[key_pos] = dot * scale;
                max_score = std::max(max_score, scores[key_pos]);
            }

            float exponential_sum = 0.0f;
            for (size_t key_pos = 0; key_pos < allowed_key_count; ++key_pos) {
                scores[key_pos] = std::exp(scores[key_pos] - max_score);
                exponential_sum += scores[key_pos];
            }
            const float inverse_sum = 1.0f / exponential_sum;

            for (size_t value_column = 0; value_column < value_dim; ++value_column) {
                float result = 0.0f;

                for (size_t key_pos = 0; key_pos < allowed_key_count; ++key_pos) {
                    const size_t v_index =
                        (key_pos * num_kv_heads + kv_head) * value_dim + value_column;
                    result += scores[key_pos] * inverse_sum
                              * llaisys::utils::cast<float>(v[v_index]);
                }

                const size_t out_index =
                    (query_pos * num_heads + query_head) * value_dim + value_column;
                out[out_index] = llaisys::utils::cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    float scale,
    size_t query_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t value_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            scale, query_len, kv_len, num_heads, num_kv_heads,
            head_dim, value_dim);
    case LLAISYS_DTYPE_F16:
        return self_attention_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            scale, query_len, kv_len, num_heads, num_kv_heads,
            head_dim, value_dim);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            scale, query_len, kv_len, num_heads, num_kv_heads,
            head_dim, value_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
