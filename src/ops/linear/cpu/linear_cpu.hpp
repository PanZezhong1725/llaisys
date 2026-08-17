#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t dtype,
            size_t batch_size,
            size_t input_size,
            size_t output_size);
} // namespace llaisys::ops::cpu
