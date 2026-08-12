#pragma once
#include "llaisys.h"
#include "../../../tensor/tensor.hpp"

namespace llaisys::ops::iluvatar {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, llaisysDataType_t type, float theta);
} // namespace llaisys::ops::iluvatar