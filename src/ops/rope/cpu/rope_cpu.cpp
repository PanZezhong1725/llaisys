#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seqlen, size_t nhead, size_t d) {
    size_t half_d = d / 2;
    for (size_t i = 0; i < seqlen; i++) {
        int64_t pos = pos_ids[i];
        for (size_t h = 0; h < nhead; h++) {
            for (size_t j = 0; j < half_d; j++) {
                float angle = static_cast<float>(pos) / std::pow(theta, 2.0f * j / d);
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                
                float a = llaisys::utils::cast<float>(in[i * nhead * d + h * d + j]);
                float b = llaisys::utils::cast<float>(in[i * nhead * d + h * d + half_d + j]);
                
                out[i * nhead * d + h * d + j] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                out[i * nhead * d + h * d + half_d + j] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
