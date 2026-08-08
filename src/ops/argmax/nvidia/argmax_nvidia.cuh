// src/ops/argmax/nvidia/argmax_nvidia.cuh
// NVIDIA CUDA implementation of argmax operator

#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel);
}
