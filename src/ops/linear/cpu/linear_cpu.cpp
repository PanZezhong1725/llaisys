#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <vector>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const float *bias,
             size_t seq_len, size_t in_features, size_t out_features, bool has_bias) {
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < out_features; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < in_features; k++) {
                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(in[i * in_features + k]);
                    b = llaisys::utils::cast<float>(weight[j * in_features + k]);
                } else {
                    a = static_cast<float>(in[i * in_features + k]);
                    b = static_cast<float>(weight[j * in_features + k]);
                }
                sum += a * b;
            }
            if (has_bias) {
                sum += bias[j];
            }
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * out_features + j] = llaisys::utils::cast<T>(sum);
            } else {
                out[i * out_features + j] = static_cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t seq_len, size_t in_features, size_t out_features,
            bool has_bias) {
    // Convert bias to float if needed (bias is stored in the same dtype as other tensors)
    std::vector<float> bias_float;
    if (has_bias) {
        bias_float.resize(out_features);
        switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            const float *bias_ptr = reinterpret_cast<const float *>(bias);
            for (size_t j = 0; j < out_features; j++) {
                bias_float[j] = bias_ptr[j];
            }
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            const llaisys::bf16_t *bias_ptr = reinterpret_cast<const llaisys::bf16_t *>(bias);
            for (size_t j = 0; j < out_features; j++) {
                bias_float[j] = llaisys::utils::cast<float>(bias_ptr[j]);
            }
            break;
        }
        case LLAISYS_DTYPE_F16: {
            const llaisys::fp16_t *bias_ptr = reinterpret_cast<const llaisys::fp16_t *>(bias);
            for (size_t j = 0; j < out_features; j++) {
                bias_float[j] = llaisys::utils::cast<float>(bias_ptr[j]);
            }
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        }
    }

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       has_bias ? bias_float.data() : nullptr,
                       seq_len, in_features, out_features, has_bias);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       has_bias ? bias_float.data() : nullptr,
                       seq_len, in_features, out_features, has_bias);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       has_bias ? bias_float.data() : nullptr,
                       seq_len, in_features, out_features, has_bias);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
