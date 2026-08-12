#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::metax {

void add(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream
);

} // namespace llaisys::ops::metax
