#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
namespace {

template <class Scalar>
void normalizeRows(
    Scalar *output,
    const Scalar *input,
    const Scalar *weight,
    size_t rows,
    size_t width,
    float epsilon) {
    for (size_t row = 0; row < rows; ++row) {
        float square_sum = 0.0f;
        for (size_t column = 0; column < width; ++column) {
            const float value = utils::cast<float>(input[row * width + column]);
            square_sum += value * value;
        }
        const float multiplier = 1.0f / std::sqrt(square_sum / static_cast<float>(width) + epsilon);
        for (size_t column = 0; column < width; ++column) {
            const float result = utils::cast<float>(input[row * width + column])
                               * multiplier
                               * utils::cast<float>(weight[column]);
            output[row * width + column] = utils::cast<Scalar>(result);
        }
    }
}

} // namespace

void rms_norm(
    std::byte *output,
    const std::byte *input,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t rows,
    size_t width,
    float epsilon) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return normalizeRows(reinterpret_cast<float *>(output), reinterpret_cast<const float *>(input), reinterpret_cast<const float *>(weight), rows, width, epsilon);
    case LLAISYS_DTYPE_F16:
        return normalizeRows(reinterpret_cast<fp16_t *>(output), reinterpret_cast<const fp16_t *>(input), reinterpret_cast<const fp16_t *>(weight), rows, width, epsilon);
    case LLAISYS_DTYPE_BF16:
        return normalizeRows(reinterpret_cast<bf16_t *>(output), reinterpret_cast<const bf16_t *>(input), reinterpret_cast<const bf16_t *>(weight), rows, width, epsilon);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
