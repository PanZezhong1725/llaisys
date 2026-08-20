#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"
#include "llaisys.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace llaisys::ops::cpu {
template <typename T>
void self_attention_impl(
	T *attn_val,
	const T *q,
	const T *k,
	const T *v,
	float scale,
	size_t q_len,
	size_t total_len,
	size_t q_heads,
  	size_t kv_heads,
  	size_t qk_dim,
  	size_t v_dim
) {
	// 每个 KV head 所对应的 Query head 的数量
	const size_t group_size = q_heads / kv_heads;

	// KV cache 中保存的长度
	const size_t past_len = total_len - q_len;

	// 为当前一个 Query head 保存它对所有 Key 位置的注意力分数
	// 对于一个固定的 Query token 和一个固定的 Query head，它最多要和 total_len 个 Key 向量分别做点积，因此会产生最多 total_len 个注意力分数
	std::vector<float> scores(total_len);

	std::vector<float> output_acc(v_dim);

	for (size_t query_idx = 0; query_idx < q_len; ++query_idx) {
		// 当前 query token 可以看到的 key 范围, causal mask
		const size_t allowed_keys = past_len + query_idx + 1;
		
		for (size_t query_head = 0; query_head < q_heads; ++query_head) {
			// 适应 MHA GQA MQA，当前 Query head 应该去读取的 KV head 的索引
			const size_t kv_head = query_head / group_size;

			// query_idx * q_heads * qk_dim + query_head * qk_dim
			const size_t q_base = (query_idx * q_heads + query_head) * qk_dim;

			// 计算 QK^T * scale
			float max_score = -std::numeric_limits<float>::infinity();
			for (size_t key_idx = 0; key_idx < allowed_keys; ++key_idx) {
				const size_t k_base = (key_idx * kv_heads + kv_head) * qk_dim;
				float dot = 0.0f;
				for (size_t dim = 0; dim < qk_dim; ++dim) {
					const float q_value = llaisys::utils::cast<float>(q[q_base + dim]);
					const float k_value = llaisys::utils::cast<float>(k[k_base + dim]);
					dot += q_value * k_value;
				}
				const float score = dot * scale;
				scores[key_idx] = score;
				max_score = std::max(max_score, score);
			}

			// stable softmax
			float denominator = 0.0f;
			for (size_t key_idx = 0; key_idx < allowed_keys; ++key_idx) {
				// 减去最大值，防止溢出；exp(89)就会爆 float32
				const float exp_score = std::exp(scores[key_idx] - max_score);
				scores[key_idx] = exp_score;
				denominator += exp_score;
			}
			const float inv_denominator = 1.0f / denominator;

			// sofatmax(A) @ V
			std::fill(output_acc.begin(), output_acc.end(), 0.0f);
			for (size_t key_idx = 0; key_idx < allowed_keys; ++key_idx) {
				const float probability = scores[key_idx] * inv_denominator;
				const size_t v_base = (key_idx * kv_heads + kv_head) * v_dim;
				for (size_t dim = 0; dim < v_dim; ++dim) {
					const float v_value = llaisys::utils::cast<float>(v[v_base + dim]);
					output_acc[dim] += probability * v_value;
				}
			}

			// 写回当前 Query head 的输出向量
			const size_t attn_val_base = (query_idx * q_heads + query_head) * v_dim;
			for (size_t dim = 0; dim < v_dim; ++dim) {
				attn_val[attn_val_base + dim] = llaisys::utils::cast<T>(output_acc[dim]);
			}	
		}
	}
}

void self_attention(
	std::byte *attn_val,
	const std::byte *q,
	const std::byte *k,
	const std::byte *v,
	float scale,
	llaisysDataType_t type,
	size_t q_len,
	size_t total_len,
	size_t q_heads,
  	size_t kv_heads,
  	size_t qk_dim,
  	size_t v_dim
) {
	switch(type) {
	case LLAISYS_DTYPE_F32:
		return self_attention_impl(
			reinterpret_cast<float *>(attn_val),
			reinterpret_cast<const float *>(q),
			reinterpret_cast<const float *>(k),
			reinterpret_cast<const float *>(v),
			scale,
			q_len,
			total_len,
			q_heads,
			kv_heads,
			qk_dim,
			v_dim
		);
	case LLAISYS_DTYPE_BF16:
		return self_attention_impl(
			reinterpret_cast<llaisys::bf16_t *>(attn_val),
			reinterpret_cast<const llaisys::bf16_t *>(q),
			reinterpret_cast<const llaisys::bf16_t *>(k),
			reinterpret_cast<const llaisys::bf16_t *>(v),
			scale,
			q_len,
			total_len,
			q_heads,
			kv_heads,
			qk_dim,
			v_dim
		);
	case LLAISYS_DTYPE_F16:
		return self_attention_impl(
			reinterpret_cast<llaisys::fp16_t *>(attn_val),
			reinterpret_cast<const llaisys::fp16_t *>(q),
			reinterpret_cast<const llaisys::fp16_t *>(k),
			reinterpret_cast<const llaisys::fp16_t *>(v),
			scale,
			q_len,
			total_len,
			q_heads,
			kv_heads,
			qk_dim,
			v_dim
		);
	default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    } 
}
} // namespace::ops::cpu