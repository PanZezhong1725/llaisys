#pragma once
#include "llaisys.h"
#include "../../../tensor/tensor.hpp"

#include <cstddef>

using tensor_t = llaisys::tensor_t;

namespace llaisys::ops::cpu {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals, llaisysDataType_t type, size_t numel);
} // namespace llaisys::ops::cpu