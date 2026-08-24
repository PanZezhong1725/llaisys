#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>

template <typename T>
void argmax_(int32_t *max_idx, float *max_val, const T *vals, size_t numel) {
    *max_idx = 0;
    *max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < numel; i++) {
        float v;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            v = llaisys::utils::cast<float>(vals[i]);
        } else {
            v = static_cast<float>(vals[i]);
        }
        if (v > *max_val) {
            *max_val = v;
            *max_idx = static_cast<int32_t>(i);
        }
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int32_t *>(max_idx), reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int32_t *>(max_idx), reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int32_t *>(max_idx), reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
