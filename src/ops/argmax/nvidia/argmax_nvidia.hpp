#pragma once
#include "llaisys.h"
#include "../../../tensor/tensor.hpp"

#include <cstddef>

namespace llaisys::ops::nvidia{
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals, llaisysDataType_t type, size_t numel);
}