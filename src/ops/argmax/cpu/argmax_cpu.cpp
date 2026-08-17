#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>

namespace llaisys::ops::cpu {
namespace {

template <class Scalar>
void findMaximum(int64_t *index, Scalar *value, const Scalar *input, size_t count) {
    size_t best = 0;
    float best_value = utils::cast<float>(input[0]);
    for (size_t i = 1; i < count; ++i) {
        const float candidate = utils::cast<float>(input[i]);
        if (candidate > best_value) {
            best = i;
            best_value = candidate;
        }
    }
    *index = static_cast<int64_t>(best);
    *value = input[best];
}

} // namespace

void argmax(std::byte *index, std::byte *value, const std::byte *input, llaisysDataType_t dtype, size_t count) {
    CHECK_ARGUMENT(count != 0, "argmax requires a non-empty input");
    auto *output_index = reinterpret_cast<int64_t *>(index);
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return findMaximum(output_index, reinterpret_cast<float *>(value), reinterpret_cast<const float *>(input), count);
    case LLAISYS_DTYPE_F16:
        return findMaximum(output_index, reinterpret_cast<fp16_t *>(value), reinterpret_cast<const fp16_t *>(input), count);
    case LLAISYS_DTYPE_BF16:
        return findMaximum(output_index, reinterpret_cast<bf16_t *>(value), reinterpret_cast<const bf16_t *>(input), count);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
