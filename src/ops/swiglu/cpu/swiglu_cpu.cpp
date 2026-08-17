#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {
namespace {

template <class Scalar>
void activate(Scalar *output, const Scalar *gate, const Scalar *up, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const float gate_value = utils::cast<float>(gate[i]);
        const float silu = gate_value / (1.0f + std::exp(-gate_value));
        output[i] = utils::cast<Scalar>(silu * utils::cast<float>(up[i]));
    }
}

} // namespace

void swiglu(std::byte *output, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t count) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return activate(reinterpret_cast<float *>(output), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), count);
    case LLAISYS_DTYPE_F16:
        return activate(reinterpret_cast<fp16_t *>(output), reinterpret_cast<const fp16_t *>(gate), reinterpret_cast<const fp16_t *>(up), count);
    case LLAISYS_DTYPE_BF16:
        return activate(reinterpret_cast<bf16_t *>(output), reinterpret_cast<const bf16_t *>(gate), reinterpret_cast<const bf16_t *>(up), count);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
