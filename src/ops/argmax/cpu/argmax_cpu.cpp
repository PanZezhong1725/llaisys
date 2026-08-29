#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    if (numel == 0) {
        max_idx[0] = 0;
        max_val[0] = llaisys::utils::cast<T>(0.0f);
        return;
    }
    size_t idx = 0;
    float max_v = llaisys::utils::cast<float>(vals[0]);
    for (size_t i = 1; i < numel; i++) {
        float v = llaisys::utils::cast<float>(vals[i]);
        if (v > max_v) {
            max_v = v;
            idx = i;
        }
    }
    max_idx[0] = static_cast<int64_t>(idx);
    max_val[0] = llaisys::utils::cast<T>(max_v);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
