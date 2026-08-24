#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::suda {
// pos_ids is a device int64 buffer of size seq_len.
// theta is the RoPE base frequency.
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, float theta,
          llaisysDataType_t dtype, size_t seq_len, size_t num_heads, size_t head_dim);
}