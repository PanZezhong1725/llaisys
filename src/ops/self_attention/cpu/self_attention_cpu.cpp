#include "self_attention_cpu.hpp"

#include "../../cpu/cpu_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

template <typename T>
void self_attention_impl(std::byte *attn_val,
                         const std::byte *q,
                         const std::byte *k,
                         const std::byte *v,
                         size_t query_length,
                         size_t key_length,
                         size_t query_heads,
                         size_t key_value_heads,
                         size_t head_dim,
                         size_t value_dim,
                         float scale) {
    const auto *query = reinterpret_cast<const T *>(q);
    const auto *key = reinterpret_cast<const T *>(k);
    const auto *value = reinterpret_cast<const T *>(v);
    auto *output = reinterpret_cast<T *>(attn_val);
    const size_t heads_per_key_value = query_heads / key_value_heads;

    std::vector<float> scores(key_length);
    for (size_t query_position = 0; query_position < query_length; ++query_position) {
        const size_t last_allowed_key = key_length - query_length + query_position;
        for (size_t query_head = 0; query_head < query_heads; ++query_head) {
            const size_t key_value_head = query_head / heads_per_key_value;
            const size_t query_offset = query_position * query_heads * head_dim + query_head * head_dim;
            for(size_t key_position = 0; key_position <= last_allowed_key; ++key_position) {
                const size_t key_offset = key_position * key_value_heads * head_dim + key_value_head * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += llaisys::ops::cpu::to_float(query[query_offset + dim]) * llaisys::ops::cpu::to_float(key[key_offset + dim]);
                }
                scores[key_position] = score * scale;
            }

            const float max_score = *std::max_element(scores.begin(),scores.end());
            float denominator = 0.0f;
            for (size_t key_position = 0; key_position <= last_allowed_key; ++key_position) {
                scores[key_position] = std::exp(scores[key_position] - max_score);
                denominator += scores[key_position];
            }

            const size_t output_offset = (query_position * query_heads * value_dim) + (query_head * value_dim);
            for (size_t dim = 0; dim < value_dim; ++dim) {
                float result = 0.0f;
                for (size_t key_position = 0; key_position <= last_allowed_key; ++key_position) {
                    const size_t value_offset = (key_position * key_value_heads * value_dim) + (key_value_head * value_dim);
                    result += scores[key_position] / denominator * llaisys::ops::cpu::to_float(value[value_offset + dim]);
            }
            output[output_offset + dim] = llaisys::ops::cpu::from_float<T>(result);
         }
    }
}
                         }
} // namespace

namespace llaisys::ops::cpu {

void self_attention(std::byte *attn_val,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t query_length,
                    size_t key_length,
                    size_t query_heads,
                    size_t key_value_heads,
                    size_t head_dim,
                    size_t value_dim,
                    float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_impl<float>(attn_val, q, k, v, query_length, key_length, query_heads, key_value_heads,
                                          head_dim, value_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_impl<llaisys::fp16_t>(attn_val, q, k, v, query_length, key_length, query_heads,
                                                    key_value_heads, head_dim, value_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_impl<llaisys::bf16_t>(attn_val, q, k, v, query_length, key_length, query_heads,
                                                    key_value_heads, head_dim, value_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
