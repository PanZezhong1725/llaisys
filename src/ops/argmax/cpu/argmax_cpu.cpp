#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

template <typename T>
void argmax_(
  std::int64_t *max_idx,
  T *max_val,
  const T *vals,
  size_t numel
) {
    size_t best_idx = 0;
    float best_val = llaisys::utils::cast<float>(vals[0]);
  
    for (size_t i = 1; i < numel; ++i) {
        const float current = llaisys::utils::cast<float>(vals[i]);

        // 当最大值出现多次时，保留第一次出现的索引
        // NaN 出现时记录第一个 NaN，之后不再更新
        if ((!std::isnan(best_val) && std::isnan(current)) || current > best_val) {
        best_idx = i;
        best_val = current;
        }

        max_idx[0] = static_cast<std::int64_t>(best_idx);
        // 直接复制原始元素，避免舍入
        max_val[0] = vals[best_idx];
    }
}

namespace llaisys::ops::cpu {
void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    size_t numel
) {
    auto *idx = reinterpret_cast<std::int64_t *>(max_idx);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(
            idx,
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            numel
        );

    case LLAISYS_DTYPE_BF16:
        return argmax_(
            idx,
            reinterpret_cast<llaisys::bf16_t *>(max_val),
            reinterpret_cast<const llaisys::bf16_t *>(vals),
            numel
        );

    case LLAISYS_DTYPE_F16:
        return argmax_(
            idx,
            reinterpret_cast<llaisys::fp16_t *>(max_val),
            reinterpret_cast<const llaisys::fp16_t *>(vals),
            numel
        );

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu