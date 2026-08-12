#include "rms_norm_cpu.hpp"

#include "../../cpu/cpu_utils.hpp"

#include <cmath>

namespace {
template <typename T>
void rms_norm_impl(std::byte *out, const std::byte *in,const std::byte *weight,size_t row_count,size_t row_size,float eps){
    const auto *input = reinterpret_cast<const T *>(in);
    const auto *weights = reinterpret_cast<const T *>(weight);
    auto *output = reinterpret_cast<T *>(out);

    for(size_t row = 0; row < row_count; ++row) {
        float sum_squ = 0.0F;
        for(size_t col = 0; col < row_size; ++col) {
            const float val = llaisys::ops::cpu::to_float(input[row * row_size + col]);
            sum_squ += val * val;
        }
        const float rms = std::sqrt(sum_squ / static_cast<float>(row_size) + eps);
        for(size_t col = 0; col < row_size; ++col) {
            const float val = llaisys::ops::cpu::to_float(input[row * row_size + col]);
            const float weight_val = llaisys::ops::cpu::to_float(weights[col]);
            output[row * row_size + col] = llaisys::ops::cpu::from_float<T>(val / rms * weight_val);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {

void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              llaisysDataType_t dtype,
              size_t row_count,
              size_t row_size,
              float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl<float>(out, in, weight, row_count, row_size, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl<llaisys::fp16_t>(out, in, weight, row_count, row_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl<llaisys::bf16_t>(out, in, weight, row_count, row_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
