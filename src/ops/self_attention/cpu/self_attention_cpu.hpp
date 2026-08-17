#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t query_length,
                    size_t key_length,
                    size_t query_heads,
                    size_t key_value_heads,
                    size_t head_dim,
                    size_t value_dim,
                    float scale);
} // namespace llaisys::ops::cpu
