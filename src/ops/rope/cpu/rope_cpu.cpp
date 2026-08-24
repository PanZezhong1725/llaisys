#include "rope_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void rope_(T *out, const T *in, const float *cos, const float *sin,
           size_t seq_len, size_t num_heads, size_t head_dim) {
    size_t half_dim = head_dim / 2;
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t h = 0; h < num_heads; h++) {
            size_t base_idx = (i * num_heads + h) * head_dim;
            for (size_t j = 0; j < half_dim; j++) {
                // x1 = in[..., j], x2 = in[..., j + half_dim]
                float x1, x2;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    x1 = llaisys::utils::cast<float>(in[base_idx + j]);
                    x2 = llaisys::utils::cast<float>(in[base_idx + j + half_dim]);
                } else {
                    x1 = static_cast<float>(in[base_idx + j]);
                    x2 = static_cast<float>(in[base_idx + j + half_dim]);
                }

                float c = cos[i * head_dim + j];
                float s = sin[i * head_dim + j];

                // output = input * cos + rotate_half(input) * sin
                // rotate_half: first half = -x2, second half = x1
                float out_first = x1 * c + (-x2) * s;
                float out_second = x2 * c + x1 * s;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[base_idx + j] = llaisys::utils::cast<T>(out_first);
                    out[base_idx + j + half_dim] = llaisys::utils::cast<T>(out_second);
                } else {
                    out[base_idx + j] = static_cast<T>(out_first);
                    out[base_idx + j + half_dim] = static_cast<T>(out_second);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *cos, const std::byte *sin,
          llaisysDataType_t dtype, size_t seq_len, size_t num_heads, size_t head_dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                     reinterpret_cast<const float *>(cos), reinterpret_cast<const float *>(sin),
                     seq_len, num_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const float *>(cos), reinterpret_cast<const float *>(sin),
                     seq_len, num_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const float *>(cos), reinterpret_cast<const float *>(sin),
                     seq_len, num_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
