#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// 使用无类型字节指针，不需要为每种 dtype 暴露一个接口
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t size);
}