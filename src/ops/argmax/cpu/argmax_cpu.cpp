#include "argmax_cpu.hpp"

#include "../../cpu/cpu_utils.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace{
    template <typename T>
    void argmax_impl(std::byte *max_idx,std::byte *max_val,const std::byte *vals,size_t numel) {
        const auto *input  = reinterpret_cast<const T *>(vals);
        auto *output_idx   = reinterpret_cast<size_t *>(max_idx);
        auto *output_val   = reinterpret_cast<T *>(max_val);

        size_t best_idx = 0;
        float best_val = llaisys::ops::cpu::to_float(input[0]);
        for (size_t i = 1; i < numel; ++i) {
            const float candidate = llaisys::ops::cpu::to_float(input[i]);
            const bool candidate_is_nan = std::isnan(candidate);
            const bool best_is_nan = std::isnan(best_val);
            if((candidate_is_nan&&!best_is_nan) || (!candidate_is_nan&&candidate>best_val&&!best_is_nan)){
                best_val = candidate;
                best_idx = i;
            }
        }
        *output_idx = static_cast<int64_t>(best_idx);
        *output_val = llaisys::ops::cpu::from_float<T>(best_val);
    }
}

namespace llaisys::ops::cpu {

void argmax(std::byte *max_idx,
            std::byte *max_val,
            const std::byte *vals,
            llaisysDataType_t dtype,
            size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_impl<float>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_F16:
        return argmax_impl<llaisys::fp16_t>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_impl<llaisys::bf16_t>(max_idx, max_val, vals, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
