#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::suda {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t seq_len, size_t in_features, size_t out_features,
            bool has_bias);
}