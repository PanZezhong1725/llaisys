#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    size_t group_size = nh / nkvh;
    std::vector<float> attn_scores(kvlen);
    
    for (size_t i = 0; i < qlen; i++) {
        for (size_t h = 0; h < nh; h++) {
            size_t kvh = h / group_size;
            
            // Compute attention scores
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < kvlen; j++) {
                float score = 0.0f;
                for (size_t d = 0; d < hd; d++) {
                    float q_val = llaisys::utils::cast<float>(q[i * nh * hd + h * hd + d]);
                    float k_val = llaisys::utils::cast<float>(k[j * nkvh * hd + kvh * hd + d]);
                    score += q_val * k_val;
                }
                score *= scale;
                attn_scores[j] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }
            
            // Apply causal mask
            for (size_t j = 0; j < kvlen; j++) {
                if (j > i + kvlen - qlen) {
                    attn_scores[j] = -std::numeric_limits<float>::infinity();
                }
            }
            
            // Find max score after masking
            max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < kvlen; j++) {
                if (attn_scores[j] > max_score) {
                    max_score = attn_scores[j];
                }
            }
            
            // Softmax
            float sum_exp = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                if (attn_scores[j] > -std::numeric_limits<float>::infinity()) {
                    attn_scores[j] = std::exp(attn_scores[j] - max_score);
                    sum_exp += attn_scores[j];
                } else {
                    attn_scores[j] = 0.0f;
                }
            }
            for (size_t j = 0; j < kvlen; j++) {
                attn_scores[j] /= sum_exp;
            }
            
            // Compute output
            for (size_t d = 0; d < hd; d++) {
                float sum = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    float v_val = llaisys::utils::cast<float>(v[j * nkvh * hd + kvh * hd + d]);
                    sum += attn_scores[j] * v_val;
                }
                attn_val[i * nh * hd + h * hd + d] = llaisys::utils::cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, llaisysDataType_t type, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), scale, qlen, kvlen, nh, nkvh, hd);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v), scale, qlen, kvlen, nh, nkvh, hd);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v), scale, qlen, kvlen, nh, nkvh, hd);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
