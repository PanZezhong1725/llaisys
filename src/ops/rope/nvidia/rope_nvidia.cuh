#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rope(std::byte *output, const std::byte *input, const std::byte *positions, llaisysDataType_t dtype, size_t sequence, size_t heads, size_t width, float theta, llaisysStream_t stream);
}
