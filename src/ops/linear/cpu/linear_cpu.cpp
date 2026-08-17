#include "linear_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {
namespace {

template <class Scalar>
void matrixProjection(
    Scalar *output,
    const Scalar *input,
    const Scalar *weight,
    const Scalar *bias,
    size_t rows,
    size_t columns,
    size_t reduction) {
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            float total = 0.0f;
            for (size_t inner = 0; inner < reduction; ++inner) {
                total += utils::cast<float>(input[row * reduction + inner])
                       * utils::cast<float>(weight[column * reduction + inner]);
            }
            if (bias != nullptr) {
                total += utils::cast<float>(bias[column]);
            }
            output[row * columns + column] = utils::cast<Scalar>(total);
        }
    }
}

} // namespace

void linear(
    std::byte *output,
    const std::byte *input,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t dtype,
    size_t rows,
    size_t columns,
    size_t reduction) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return matrixProjection(
            reinterpret_cast<float *>(output), reinterpret_cast<const float *>(input),
            reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias),
            rows, columns, reduction);
    case LLAISYS_DTYPE_F16:
        return matrixProjection(
            reinterpret_cast<fp16_t *>(output), reinterpret_cast<const fp16_t *>(input),
            reinterpret_cast<const fp16_t *>(weight), reinterpret_cast<const fp16_t *>(bias),
            rows, columns, reduction);
    case LLAISYS_DTYPE_BF16:
        return matrixProjection(
            reinterpret_cast<bf16_t *>(output), reinterpret_cast<const bf16_t *>(input),
            reinterpret_cast<const bf16_t *>(weight), reinterpret_cast<const bf16_t *>(bias),
            rows, columns, reduction);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
