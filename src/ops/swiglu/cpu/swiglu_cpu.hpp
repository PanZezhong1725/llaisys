#pragma once
#include "llaisys.h"
#include "../../../tensor/tensor.hpp"

using tensor_t = llaisys::tensor_t;

namespace llaisys::ops::cpu {
void swiglu(tensor_t out, tensor_t gate, tensor_t up, llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::cpu