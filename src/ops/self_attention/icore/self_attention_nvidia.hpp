#pragma once
#include "llaisys.h"
#include "../../../tensor/tensor.hpp"

#include <cstddef>

using tensor_t = llaisys::tensor_t;

namespace llaisys::ops::iluvatar {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v,
                    float scale, llaisysDataType_t type);
} // namespace llaisys::ops::iluvatar
