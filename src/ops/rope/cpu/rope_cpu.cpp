#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

template <typename T>
void rope_(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    float theta,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim) {
    const size_t half_dim = head_dim / 2;
    std::vector<float> frequency_denominators(half_dim);

    for (size_t pair = 0; pair < half_dim; ++pair) {
        const float exponent =
            2.0f * static_cast<float>(pair) / static_cast<float>(head_dim);
        frequency_denominators[pair] = std::pow(theta, exponent);
    }

    for (size_t seq = 0; seq < seq_len; ++seq) {
        const float position = static_cast<float>(pos_ids[seq]);

        for (size_t pair = 0; pair < half_dim; ++pair) {
            const float angle = position / frequency_denominators[pair];
            const float sine = std::sin(angle);
            const float cosine = std::cos(angle);

            for (size_t head = 0; head < num_heads; ++head) {
                const size_t base = (seq * num_heads + head) * head_dim;
                const size_t first = base + pair;
                const size_t second = base + half_dim + pair;
                const float a = llaisys::utils::cast<float>(in[first]);
                const float b = llaisys::utils::cast<float>(in[second]);

                out[first] = llaisys::utils::cast<T>(a * cosine - b * sine);
                out[second] = llaisys::utils::cast<T>(b * cosine + a * sine);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    llaisysDataType_t type,
    float theta,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim) {
    const auto *positions = reinterpret_cast<const int64_t *>(pos_ids);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            positions, theta, seq_len, num_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            positions, theta, seq_len, num_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            positions, theta, seq_len, num_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
