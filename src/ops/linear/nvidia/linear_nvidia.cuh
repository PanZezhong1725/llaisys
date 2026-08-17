#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void linear(std::byte *output, const std::byte *input, const std::byte *weight, const std::byte *bias, llaisysDataType_t dtype, size_t rows, size_t columns, size_t reduction, llaisysStream_t stream);
}
