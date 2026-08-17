#pragma once
#include "llaisys.h"
#include "../../../tensor/tensor.hpp"

using tensor_t = llaisys::tensor_t;

namespace llaisys::ops::cpu {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, llaisysDataType_t type, float eps);
} // namespace llaisys::ops::cpu