#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
/*
 *  M：输入行数，例如 batch/token 数
 *  K：输入特征数
 *  N：输出特征数
 *  
 *  in     [M, K]
 *  weight [N, K]
 *  out    [M, N]
 *  bias   [N] 或 nullptr
*/ 
void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t m,
    size_t n,
    size_t k 
);

}