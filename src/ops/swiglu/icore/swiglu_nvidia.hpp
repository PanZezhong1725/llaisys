#pragma once
#include "llaisys.h"

#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::iluvatar {
void swiglu(tensor_t out, tensor_t gate, tensor_t up, llaisysDataType_t dtype, size_t numel);
} // namespace llaisys::ops::iluvatar