// src/ops/rms_norm/nvidia/rms_norm_nvidia.cuh
#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t rows, size_t cols);
}
