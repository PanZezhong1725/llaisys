#pragma once
#include "llaisys.h"
#include "../../../tensor/tensor.hpp"

using tensor_t = llaisys::tensor_t;

namespace llaisys::ops::cpu {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, llaisysDataType_t type, float theta);
} // namespace llaisys::ops::cpu