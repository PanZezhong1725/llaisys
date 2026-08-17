#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void embedding(std::byte *output, const std::byte *indices, const std::byte *table, llaisysDataType_t dtype, size_t index_count, size_t row_count, size_t row_width, llaisysStream_t stream);
}
