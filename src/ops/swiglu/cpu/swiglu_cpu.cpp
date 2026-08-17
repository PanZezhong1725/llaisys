#include "swiglu_cpu.hpp"

#include "../../cpu/cpu_utils.hpp"

#include <cmath>

namespace {

template <typename T>
void swiglu_impl(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel) {
    const auto *gate_values = reinterpret_cast<const T *>(gate);
    const auto *up_values = reinterpret_cast<const T *>(up);
    auto *output = reinterpret_cast<T *>(out);

    for (size_t i = 0; i < numel; ++i) {
        const float gate_value = llaisys::ops::cpu::to_float(gate_values[i]);
        const float up_value = llaisys::ops::cpu::to_float(up_values[i]);
        const float sigmoid_gate = 1.0F / (1.0F + std::exp(-gate_value));
        output[i] = llaisys::ops::cpu::from_float<T>(up_value * gate_value * sigmoid_gate);
    }
}

} // namespace

namespace llaisys::ops::cpu {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_impl<llaisys::fp16_t>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl<llaisys::bf16_t>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
