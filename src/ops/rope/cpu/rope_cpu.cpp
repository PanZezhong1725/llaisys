#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
// a' = a*cos(phi) - b*sin(phi)，b' = b*cos(phi) + a*sin(phi)，(a,b) = (in[j], in[j+d/2])
// phi = pos_ids[i] / theta^(2j/d), j = 0 .. d/2-1
template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seqlen, size_t nhead, size_t d) {
    for(size_t i = 0; i < seqlen; i++) {
        for(size_t h = 0; h < nhead; h++) {
            size_t base = i * nhead * d + h * d;
            for(size_t j = 0; j < d / 2; j++) {
                float phi = pos_ids[i] / std::pow(theta, 2.0f * j / d);
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);
                float a = llaisys::utils::cast<float>(in[base +  j]);
                float b = llaisys::utils::cast<float>(in[base + j + d/2]);
                out[base + j] = llaisys::utils::cast<T>(a * cos_phi - b * sin_phi);
                out[base + j + d/2] = llaisys::utils::cast<T>(b * cos_phi + a * sin_phi);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d, float theta) {
    const int64_t *pos = reinterpret_cast<const int64_t *>(pos_ids);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pos, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), pos, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), pos, theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} 
