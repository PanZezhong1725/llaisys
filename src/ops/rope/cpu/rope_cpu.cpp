#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const std::int64_t *pos_ids,
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t half_dim = head_dim / 2;
    size_t head_size = head_dim;
    size_t row_size = n_heads * head_dim;
    
    for (size_t i = 0; i < seq_len; i++) {
        std::int64_t pos_id = pos_ids[i];
        float pos = static_cast<float>(pos_id);
        
        for (size_t h = 0; h < n_heads; h++) {
            size_t row_start = i * row_size + h * head_size;

            for (size_t j = 0; j < half_dim; j++) {
                float freq = pos / std::pow(theta, (2.0f * static_cast<float>(j)) / static_cast<float>(head_dim));
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                size_t a_idx = row_start + j;
                size_t b_idx = row_start + j + half_dim;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    float a = llaisys::utils::cast<float>(in[a_idx]);
                    float b = llaisys::utils::cast<float>(in[b_idx]);
                    out[a_idx] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                    out[b_idx] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
                } else {
                    out[a_idx] = in[a_idx] * cos_val - in[b_idx] * sin_val;
                    out[b_idx] = in[b_idx] * cos_val + in[a_idx] * sin_val;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, llaisysDataType_t type, float theta) {
    const auto &shape = out->shape();
    ASSERT(shape.size() == 3, "RoPE expects 3D tensor [seq_len, n_heads, head_dim]");
    size_t seq_len = shape[0];
    size_t n_heads = shape[1];
    size_t head_dim = shape[2];
    ASSERT(head_dim % 2 == 0, "Head dimension must be even for RoPE.");

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out->data()),
                     reinterpret_cast<const float *>(in->data()),
                     reinterpret_cast<const std::int64_t *>(pos_ids->data()),
                     seq_len, n_heads, head_dim,
                     theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                     reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                     reinterpret_cast<const std::int64_t *>(pos_ids->data()),
                     seq_len, n_heads, head_dim,
                     theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                     reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                     reinterpret_cast<const std::int64_t *>(pos_ids->data()),
                     seq_len, n_heads, head_dim,
                     theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
