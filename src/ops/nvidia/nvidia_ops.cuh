#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void add(std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t);
void argmax(std::byte *, std::byte *, const std::byte *, llaisysDataType_t, size_t);
void embedding(std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t, size_t, size_t);
void linear(std::byte *, const std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t, size_t, size_t);
void rms_norm(std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t, size_t, float);
void rope(std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t, size_t, size_t, float);
void self_attention(std::byte *, const std::byte *, const std::byte *, const std::byte *, llaisysDataType_t,
                    size_t, size_t, size_t, size_t, size_t, size_t, float);
void swiglu(std::byte *, const std::byte *, const std::byte *, llaisysDataType_t, size_t);
} // namespace llaisys::ops::nvidia
