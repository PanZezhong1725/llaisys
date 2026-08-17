#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              llaisysDataType_t dtype,
              size_t row_count,
              size_t row_size,
              float eps);
} // namespace llaisys::ops::cpu
