#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        const float gate_value = llaisys::utils::cast<float>(gate[i]);
        const float up_value = llaisys::utils::cast<float>(up[i]);
        const float silu = gate_value / (1.0f + std::exp(-gate_value));
        out[i] = llaisys::utils::cast<T>(up_value * silu);
    }
}

namespace llaisys::ops::cpu {
void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t type,
    size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(gate),
            reinterpret_cast<const llaisys::fp16_t *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(gate),
            reinterpret_cast<const llaisys::bf16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
