#include "self_attention.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <limits>
#include <vector>

// V4: 在 V3 基础上加上 softmax 数值稳定处理。原来的单遍循环(打分+exp+累加一起做)拆成两遍:
// 第一遍算出所有 score[j] 和它们的最大值 max_score;第二遍用 exp(score[j] - max_score) 代替
// exp(score[j]),这样指数的输入永远 <= 0,不会溢出。两遍之间用 std::vector<float> scores 传递
// 打分结果,避免重复计算点积。
// q        : [seqlen,    nhead,   d ]
// k        : [total_len, nkvhead, d ]
// qk^T     : [seqlen,    nhead,   total_len]
// softmax(scale * qk^T) : [seqlen, nhead, total_len]
// v        : [total_len, nkvhead, dv]
// attn_val : [seqlen,    nhead,   dv]
// attn_val = softmax(scale * q * k^T) * v
template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                      size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv,
                      float scale) {
    size_t causal_offset = total_len - seqlen; // 位置 i 能看到的 key 是 j = 0..(i + causal_offset)
    size_t group = nhead / nkvhead;            // 每 group 个 query head 共享 1 个 kv head

    for (size_t i = 0; i < seqlen; i++) {    // 遍历每个 query token
        for (size_t h = 0; h < nhead; h++) { // 遍历每个 query head
            size_t kvh = h / group;          // 该 query head 对应的 kv head
            size_t limit = i + causal_offset; // 因果掩码:j 的范围是 [0, limit](闭区间)

            // 第一遍:算出所有 score[j] 并记录最大值 max_score。
            std::vector<float> scores(limit + 1);
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j <= limit; j++) {
                float score = 0.0f;
                for (size_t dim = 0; dim < d; dim++) {
                    score += llaisys::utils::cast<float>(q[i * nhead * d + h * d + dim]) * llaisys::utils::cast<float>(k[j * nkvhead * d + kvh * d + dim]);
                }
                score *= scale;
                scores[j] = score;
                max_score = std::max(max_score, score);
            }

            float sum_exp = 0.0f;
            std::vector<float> acc(dv, 0.0f);
            // 第二遍:用 exp(score - max_score) 做数值稳定的 softmax 加权求和。
            for (size_t j = 0; j <= limit; j++) {
                float e = std::exp(scores[j] - max_score);
                sum_exp += e;
                for (size_t t = 0; t < dv; t++) {
                    acc[t] += e * llaisys::utils::cast<float>(v[j * nkvhead * dv + kvh * dv + t]);
                }
            }
            // 归一化并写回:attn_val[i,h,:] = acc[:] / sum_exp
            for (size_t t = 0; t < dv; t++) {
                attn_val[i * nhead * dv + h * dv + t] = llaisys::utils::cast<T>(acc[t] / sum_exp);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                     llaisysDataType_t type,
                     size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv,
                     float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                               seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                               seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                               seqlen, total_len, nhead, nkvhead, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
