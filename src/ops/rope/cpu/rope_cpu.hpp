#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *cos, const std::byte *sin,
          llaisysDataType_t dtype, size_t seq_len, size_t num_heads, size_t head_dim);
}
