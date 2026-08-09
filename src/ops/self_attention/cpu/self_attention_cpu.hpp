#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    float scale,
    size_t query_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t value_dim);
}
