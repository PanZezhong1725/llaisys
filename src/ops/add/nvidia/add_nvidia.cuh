#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void add(std::byte *output, const std::byte *left, const std::byte *right, llaisysDataType_t dtype, size_t count, llaisysStream_t stream);
}
