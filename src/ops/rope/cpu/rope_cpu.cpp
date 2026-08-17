#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

namespace llaisys::ops::cpu {
namespace {

template <class Scalar>
void rotate(
    Scalar *output,
    const Scalar *input,
    const int64_t *positions,
    size_t sequence,
    size_t heads,
    size_t width,
    float theta) {
    const size_t half = width / 2;
    for (size_t token = 0; token < sequence; ++token) {
        for (size_t head = 0; head < heads; ++head) {
            const size_t base = (token * heads + head) * width;
            for (size_t pair = 0; pair < half; ++pair) {
                const float exponent = 2.0f * static_cast<float>(pair) / static_cast<float>(width);
                const float angle = static_cast<float>(positions[token]) / std::pow(theta, exponent);
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);
                const float left = utils::cast<float>(input[base + pair]);
                const float right = utils::cast<float>(input[base + half + pair]);
                output[base + pair] = utils::cast<Scalar>(left * cosine - right * sine);
                output[base + half + pair] = utils::cast<Scalar>(right * cosine + left * sine);
            }
        }
    }
}

} // namespace

void rope(
    std::byte *output,
    const std::byte *input,
    const std::byte *positions,
    llaisysDataType_t dtype,
    size_t sequence,
    size_t heads,
    size_t width,
    float theta) {
    const auto *position_values = reinterpret_cast<const int64_t *>(positions);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rotate(reinterpret_cast<float *>(output), reinterpret_cast<const float *>(input), position_values, sequence, heads, width, theta);
    case LLAISYS_DTYPE_F16:
        return rotate(reinterpret_cast<fp16_t *>(output), reinterpret_cast<const fp16_t *>(input), position_values, sequence, heads, width, theta);
    case LLAISYS_DTYPE_BF16:
        return rotate(reinterpret_cast<bf16_t *>(output), reinterpret_cast<const bf16_t *>(input), position_values, sequence, heads, width, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
