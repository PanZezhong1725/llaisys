#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(
    std::byte *output,
    const std::byte *input,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t rows,
    size_t width,
    float epsilon);
}
