#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out,
               const std::byte *index,
               const std::byte *weight,
               llaisysDataType_t dtype,
               size_t index_count,
               size_t vocabulary_size,
               size_t embedding_dim,
               size_t element_size);
} // namespace llaisys::ops::cpu
