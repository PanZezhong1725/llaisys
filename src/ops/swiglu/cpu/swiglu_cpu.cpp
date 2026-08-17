#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_cpu(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float s = g / (1.0f + std::exp(-g)); // sigmoid
        out[i] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(up[i]) * s);
    }
}

namespace llaisys::ops::cpu {
void swiglu(tensor_t out, tensor_t gate, tensor_t up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_cpu(reinterpret_cast<float *>(out->data()),
                          reinterpret_cast<const float *>(gate->data()),
                          reinterpret_cast<const float *>(up->data()),
                          numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_cpu(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                          reinterpret_cast<const llaisys::bf16_t *>(gate->data()),
                          reinterpret_cast<const llaisys::bf16_t *>(up->data()),
                          numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_cpu(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                          reinterpret_cast<const llaisys::fp16_t *>(gate->data()),
                          reinterpret_cast<const llaisys::fp16_t *>(up->data()),
                          numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu