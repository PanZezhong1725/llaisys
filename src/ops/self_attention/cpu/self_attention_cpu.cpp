#include "self_attention_cpu.hpp"
#include "../../../utils/matmul_cpu.hpp"

#include <cmath>
#include <vector>

// 自注意力: attn_val = softmax(Q·K^T·scale + causal_mask)·V
// q: (qlen, nh, hd)  k/v: (kvlen, nkvh, hd)  attn_val: (qlen, nh, hd)
template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    size_t group_size = nh / nkvh;
    // 因果掩码
    size_t diag = kvlen - qlen;

    std::vector<float> scores(kvlen);

    for (size_t h = 0; h < nh; ++h) {
        size_t kv_head = h / group_size;

        for (size_t qpos = 0; qpos < qlen; ++qpos) {
            // Step 1: Q[h][qpos] · K[kv_head][:] * scale
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t kpos = 0; kpos < kvlen; ++kpos) {
                float acc = llaisys::ops::cpu::dot_product(
                    &q[qpos * nh * hd + h * hd],
                    &k[kpos * nkvh * hd + kv_head * hd],
                    hd) * scale;
                if (kpos > qpos + diag) acc = -std::numeric_limits<float>::infinity();
                scores[kpos] = acc;
                if (acc > max_score) max_score = acc;
            }

            // Step 2: Softmax
            float sum_exp = 0.0f;
            for (size_t kpos = 0; kpos < kvlen; ++kpos) {
                if (scores[kpos] == -std::numeric_limits<float>::infinity()) {
                    scores[kpos] = 0.0f;
                } else {
                    scores[kpos] = std::exp(scores[kpos] - max_score);
                    sum_exp += scores[kpos];
                }
            }
            float inv_sum = 1.0f / sum_exp;
            for (size_t kpos = 0; kpos < kvlen; ++kpos) scores[kpos] *= inv_sum;

            // Step 3: scores · V[kv_head]
            for (size_t d = 0; d < hd; ++d) {
                float out_val = 0.0f;
                for (size_t kpos = 0; kpos < kvlen; ++kpos) {
                    out_val += scores[kpos] * llaisys::utils::cast<float>(
                        v[kpos * nkvh * hd + kv_head * hd + d]);
                }
                attn_val[qpos * nh * hd + h * hd + d] = llaisys::utils::cast<T>(out_val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale, llaisysDataType_t type) {
    const auto &q_shape = q->shape();
    const auto &k_shape = k->shape();
    const size_t qlen = q_shape[0];
    const size_t nh   = q_shape[1];
    const size_t hd   = q_shape[2];
    const size_t kvlen = k_shape[0];
    const size_t nkvh  = k_shape[1];

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val->data()), 
                        reinterpret_cast<const float *>(q->data()), 
                        reinterpret_cast<const float *>(k->data()), 
                        reinterpret_cast<const float *>(v->data()), 
                        scale, qlen, kvlen, nh, nkvh, hd);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val->data()), 
                        reinterpret_cast<const llaisys::bf16_t *>(q->data()), 
                        reinterpret_cast<const llaisys::bf16_t *>(k->data()), 
                        reinterpret_cast<const llaisys::bf16_t *>(v->data()), 
                        scale, qlen, kvlen, nh, nkvh, hd);    
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val->data()), 
                        reinterpret_cast<const llaisys::fp16_t *>(q->data()), 
                        reinterpret_cast<const llaisys::fp16_t *>(k->data()), 
                        reinterpret_cast<const llaisys::fp16_t *>(v->data()), 
                        scale, qlen, kvlen, nh, nkvh, hd);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu