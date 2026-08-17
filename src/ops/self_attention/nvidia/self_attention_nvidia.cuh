#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void self_attention(std::byte *output, const std::byte *query, const std::byte *key, const std::byte *value, llaisysDataType_t dtype, size_t query_length, size_t key_length, size_t query_heads, size_t key_value_heads, size_t query_key_width, size_t value_width, float scale, llaisysStream_t stream);
}
