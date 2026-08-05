#pragma once
#include "llaisys.h"
#include "../../../device/device_resource.hpp"
#include <cstddef>
// in :[M,K], weight :[N,K], bias :[N], out :[M,N]
// M: in's first dimension, N: weight's first dimension, K: in's second dimension
// out = in * weight^T + bias
namespace llaisys::ops::iluvatar {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
     llaisysDataType_t type, size_t M, size_t N, size_t K,llaisys::device::DeviceResource *resource);
}
