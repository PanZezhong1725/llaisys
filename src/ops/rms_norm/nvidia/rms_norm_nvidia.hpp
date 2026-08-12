#pragma once
#include "llaisys.h"

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::nvidia {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, llaisysDataType_t dtype, float eps);
} // namespace llaisys::ops::nvidia