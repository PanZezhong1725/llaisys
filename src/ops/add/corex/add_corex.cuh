#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::corex {
void add(std::byte *out, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel);
}
