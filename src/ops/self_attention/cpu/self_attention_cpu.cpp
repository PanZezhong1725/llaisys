#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops::cpu {
namespace {

template <class Scalar>
void causalAttention(
    Scalar *output,
    const Scalar *query,
    const Scalar *key,
    const Scalar *value,
    size_t query_length,
    size_t key_length,
    size_t query_heads,
    size_t key_value_heads,
    size_t query_key_width,
    size_t value_width,
    float scale) {
    const size_t heads_per_group = query_heads / key_value_heads;
    std::vector<float> probabilities(key_length);

    for (size_t token = 0; token < query_length; ++token) {
        const size_t allowed = key_length - query_length + token + 1;
        for (size_t head = 0; head < query_heads; ++head) {
            const size_t kv_head = head / heads_per_group;
            float largest = -std::numeric_limits<float>::infinity();

            for (size_t source = 0; source < allowed; ++source) {
                float score = 0.0f;
                const size_t q_base = (token * query_heads + head) * query_key_width;
                const size_t k_base = (source * key_value_heads + kv_head) * query_key_width;
                for (size_t component = 0; component < query_key_width; ++component) {
                    score += utils::cast<float>(query[q_base + component])
                           * utils::cast<float>(key[k_base + component]);
                }
                score *= scale;
                probabilities[source] = score;
                largest = std::max(largest, score);
            }

            float normalizer = 0.0f;
            for (size_t source = 0; source < allowed; ++source) {
                probabilities[source] = std::exp(probabilities[source] - largest);
                normalizer += probabilities[source];
            }

            for (size_t component = 0; component < value_width; ++component) {
                float aggregate = 0.0f;
                for (size_t source = 0; source < allowed; ++source) {
                    const size_t v_base = (source * key_value_heads + kv_head) * value_width;
                    aggregate += (probabilities[source] / normalizer)
                               * utils::cast<float>(value[v_base + component]);
                }
                output[(token * query_heads + head) * value_width + component] = utils::cast<Scalar>(aggregate);
            }
        }
    }
}

} // namespace

void self_attention(
    std::byte *output,
    const std::byte *query,
    const std::byte *key,
    const std::byte *value,
    llaisysDataType_t dtype,
    size_t query_length,
    size_t key_length,
    size_t query_heads,
    size_t key_value_heads,
    size_t query_key_width,
    size_t value_width,
    float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return causalAttention(reinterpret_cast<float *>(output), reinterpret_cast<const float *>(query), reinterpret_cast<const float *>(key), reinterpret_cast<const float *>(value), query_length, key_length, query_heads, key_value_heads, query_key_width, value_width, scale);
    case LLAISYS_DTYPE_F16:
        return causalAttention(reinterpret_cast<fp16_t *>(output), reinterpret_cast<const fp16_t *>(query), reinterpret_cast<const fp16_t *>(key), reinterpret_cast<const fp16_t *>(value), query_length, key_length, query_heads, key_value_heads, query_key_width, value_width, scale);
    case LLAISYS_DTYPE_BF16:
        return causalAttention(reinterpret_cast<bf16_t *>(output), reinterpret_cast<const bf16_t *>(query), reinterpret_cast<const bf16_t *>(key), reinterpret_cast<const bf16_t *>(value), query_length, key_length, query_heads, key_value_heads, query_key_width, value_width, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
