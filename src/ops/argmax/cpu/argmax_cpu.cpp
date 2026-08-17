#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <type_traits>

template <typename T>
void argmax_(const T *vals, T *max_val, std::int64_t *max_idx, size_t numel) {
    if (numel == 0) {
        return;
    }

    max_val[0] = vals[0];
    max_idx[0] = 0;

    for (size_t i = 1; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            if (llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(max_val[0])) {
                max_val[0] = vals[i];
                max_idx[0] = static_cast<std::int64_t>(i);
            }
        } else {
            if (vals[i] > max_val[0]) {
                max_val[0] = vals[i];
                max_idx[0] = static_cast<std::int64_t>(i);
            }
        }
    }
}

// 获取张量`vals`的最大值及其索引，并分别存储在`max_val`和`max_idx`中。
namespace llaisys::ops::cpu {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<const float *>(vals->data()), 
                       reinterpret_cast<float *>(max_val->data()), 
                       reinterpret_cast<std::int64_t *>(max_idx->data()), 
                       numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<const llaisys::bf16_t *>(vals->data()), 
                       reinterpret_cast<llaisys::bf16_t *>(max_val->data()), 
                       reinterpret_cast<std::int64_t *>(max_idx->data()), 
                       numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<const llaisys::fp16_t *>(vals->data()), 
                       reinterpret_cast<llaisys::fp16_t *>(max_val->data()), 
                       reinterpret_cast<std::int64_t *>(max_idx->data()), 
                       numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

}
} // namespace llaisys::ops::cpu