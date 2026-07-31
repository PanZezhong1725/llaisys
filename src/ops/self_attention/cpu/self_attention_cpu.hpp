#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
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
);
}