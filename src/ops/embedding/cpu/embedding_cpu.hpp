#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    size_t token_count,
    size_t vocab_size,
    size_t hidden_size,
    size_t element_size);
}
