#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(std::byte *index, std::byte *value, const std::byte *input, llaisysDataType_t dtype, size_t count);
}
