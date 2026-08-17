#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void swiglu(std::byte *output, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t count, llaisysStream_t stream);
}
