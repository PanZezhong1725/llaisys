#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t seq_len, size_t hidden_size, float eps) {
    for (size_t i = 0; i < seq_len; i++) {
        // Compute mean of x^2
        float sum_sq = 0.0f;
        for (size_t j = 0; j < hidden_size; j++) {
            float v;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                v = llaisys::utils::cast<float>(in[i * hidden_size + j]);
            } else {
                v = static_cast<float>(in[i * hidden_size + j]);
            }
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);

        // Normalize and multiply by weight
        for (size_t j = 0; j < hidden_size; j++) {
            float v;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                v = llaisys::utils::cast<float>(in[i * hidden_size + j]);
            } else {
                v = static_cast<float>(in[i * hidden_size + j]);
            }
            float w;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                w = llaisys::utils::cast<float>(weight[j]);
            } else {
                w = static_cast<float>(weight[j]);
            }
            float result = (v / rms) * w;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * hidden_size + j] = llaisys::utils::cast<T>(result);
            } else {
                out[i * hidden_size + j] = static_cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t dtype, size_t seq_len, size_t hidden_size, float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), seq_len, hidden_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), seq_len, hidden_size, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), seq_len, hidden_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
