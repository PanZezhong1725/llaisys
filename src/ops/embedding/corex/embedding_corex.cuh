#pragma once

#include <cstddef>

namespace llaisys::ops::corex {
void embedding(std::byte *out, const std::byte *index,
               const std::byte *weight, size_t token_count,
               size_t vocab_size, size_t hidden_size, size_t element_size);
}
