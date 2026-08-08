// src/ops/rearrange/nvidia/rearrange_nvidia.cuh
#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, size_t numel);
}
