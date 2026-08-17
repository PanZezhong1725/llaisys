#pragma once
#include "llaisys.h"
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::iluvatar {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias, llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::iluvatar
