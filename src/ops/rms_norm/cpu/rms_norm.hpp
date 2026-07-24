#pragma once
#include "llaisys.h"

#include <cstddef>
// out[i][j] = weight[j] * in[i][j] / sqrt(mean_j(in[i][j]^2) + eps)
// in :[rows, d], weight :[d], out :[rows, d]
namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t type, size_t rows, size_t d);
}
