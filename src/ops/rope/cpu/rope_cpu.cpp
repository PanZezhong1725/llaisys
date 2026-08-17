#include "rope_cpu.hpp"

#include "../../cpu/cpu_utils.hpp"

#include <cmath>
#include <cstdint>

namespace {

template <typename T>
void rope_impl(std::byte *out,const std::byte *in,const std::byte *pos_ids,size_t sequence_length,size_t head_count,size_t head_dim,float theta) {
    const auto *input = reinterpret_cast<const T *>(in);
    const auto *positions = reinterpret_cast<const int64_t *>(pos_ids);
    auto *output = reinterpret_cast<T *>(out);
    const size_t half_dim = head_dim / 2;

    for (size_t seq = 0; seq < sequence_length; ++seq) {
        for(size_t head = 0; head < head_count; ++head) {
            const size_t base = seq * head_count * head_dim + head * head_dim;
            for (size_t dim = 0; dim < half_dim; ++dim) {
                const float exponent = static_cast<float>(2*dim)/static_cast<float>(head_dim);
                const float angle = static_cast<float>(positions[seq]) / std::pow(theta, exponent);
                const float sine = std::sin(angle);
                const float cosine = std::cos(angle);
                const float a = llaisys::ops::cpu::to_float(input[base + dim]);
                const float b = llaisys::ops::cpu::to_float(input[base + dim + half_dim]);
                output[base + dim] = llaisys::ops::cpu::from_float<T>(a * cosine - b * sine);
                output[base + dim + half_dim] = llaisys::ops::cpu::from_float<T>(a * sine + b * cosine);
            }
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {

void rope(std::byte *out,
          const std::byte *in,
          const std::byte *pos_ids,
          llaisysDataType_t dtype,
          size_t sequence_length,
          size_t head_count,
          size_t head_dim,
          float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_impl<float>(out, in, pos_ids, sequence_length, head_count, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_impl<llaisys::fp16_t>(out, in, pos_ids, sequence_length, head_count, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_impl<llaisys::bf16_t>(out, in, pos_ids, sequence_length, head_count, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
