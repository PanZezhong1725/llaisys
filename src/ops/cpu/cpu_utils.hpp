#pragma once

#include "../../utils.hpp"

namespace llaisys::ops::cpu {

inline bool is_supported_float(llaisysDataType_t dtype) {
    return dtype == LLAISYS_DTYPE_F32 || dtype == LLAISYS_DTYPE_F16 || dtype == LLAISYS_DTYPE_BF16;
}

template <typename T>
float to_float(T value) {
    return llaisys::utils::cast<float>(value);
}

template <typename T>
T from_float(float value) {
    return llaisys::utils::cast<T>(value);
}

} // namespace llaisys::ops::cpu
