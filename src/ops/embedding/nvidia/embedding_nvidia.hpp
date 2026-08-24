#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *indices, const std::byte *weight,
               llaisysDataType_t dtype, size_t seq_len, size_t embed_dim, size_t vocab_size);
}
