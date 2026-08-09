#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    T max_value = vals[0];
    float max_comparable = llaisys::utils::cast<float>(max_value);
    int64_t max_index = 0;
    for (size_t i = 1; i < numel; i++) {
        float current_comparable = llaisys::utils::cast<float>(vals[i]);
        if (current_comparable > max_comparable) {
            max_comparable = current_comparable;
            max_value = vals[i];
            max_index = static_cast<int64_t>(i);
        }
    }
    max_val[0] = max_value;
    max_idx[0] = max_index;
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu