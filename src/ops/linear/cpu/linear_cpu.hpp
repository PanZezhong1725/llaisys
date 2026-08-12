#pragma once
#include "llaisys.h"
#include "../../../tensor/tensor.hpp"

using tensor_t = llaisys::tensor_t;

namespace llaisys::ops::cpu {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias, llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::cpu