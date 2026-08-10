#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

namespace llaisys::ops::cpu {

static inline float to_float(const std::byte *ptr, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return *reinterpret_cast<const float *>(ptr);
    case LLAISYS_DTYPE_F16:
        return llaisys::utils::cast<float>(*reinterpret_cast<const llaisys::fp16_t *>(ptr));
    case LLAISYS_DTYPE_BF16:
        return llaisys::utils::cast<float>(*reinterpret_cast<const llaisys::bf16_t *>(ptr));
    default:
        return 0.0f;
    }
}

static inline void from_float(float val, std::byte *ptr, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        *reinterpret_cast<float *>(ptr) = val;
        break;
    case LLAISYS_DTYPE_F16:
        *reinterpret_cast<llaisys::fp16_t *>(ptr) = llaisys::utils::cast<llaisys::fp16_t>(val);
        break;
    case LLAISYS_DTYPE_BF16:
        *reinterpret_cast<llaisys::bf16_t *>(ptr) = llaisys::utils::cast<llaisys::bf16_t>(val);
        break;
    default:
        break;
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t seq_len, size_t hidden_size) {
    // output = silu(gate) * up
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

    size_t elem_size = 0;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        elem_size = 4;
        break;
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        elem_size = 2;
        break;
    default:
        return;
    }

    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < hidden_size; j++) {
            size_t idx = i * hidden_size + j;
            float g = to_float(gate + idx * elem_size, dtype);
            float u = to_float(up + idx * elem_size, dtype);
            // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
            float silu_g = g / (1.0f + std::exp(-g));
            float result = silu_g * u;
            from_float(result, out + idx * elem_size, dtype);
        }
    }
}

} // namespace llaisys::ops::cpu
