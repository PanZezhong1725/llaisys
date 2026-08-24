#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::suda {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t seq_len, size_t kv_len, size_t num_heads,
                    size_t num_kv_heads, size_t head_dim, float scale);

}