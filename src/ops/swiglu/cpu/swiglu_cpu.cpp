#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t rows, size_t d) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < d; j++) {
            float gate_val = llaisys::utils::cast<float>(gate[i * d + j]);
            gate_val = gate_val / (1 + std::exp(-gate_val)); // silu(gate)
            float up_val = llaisys::utils::cast<float>(up[i * d + j]);
            float swiglu_val = gate_val * up_val;
            out[i * d + j] = llaisys::utils::cast<T>(swiglu_val);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t rows, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up), rows, d);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up), rows, d);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up), rows, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
