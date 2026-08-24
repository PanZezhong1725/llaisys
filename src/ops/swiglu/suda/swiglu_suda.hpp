#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::suda {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t seq_len, size_t hidden_size);
}