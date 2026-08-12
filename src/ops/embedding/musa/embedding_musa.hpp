#pragma once
#include "llaisys.h"

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::musa {
void embedding(tensor_t out, tensor_t index, tensor_t weight, llaisysDataType_t type, size_t num_rows, size_t num_cols);
} // namespace llaisys::ops::musa
