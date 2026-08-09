#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::corex {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, float eps, size_t rows,
              size_t hidden_size);
}
