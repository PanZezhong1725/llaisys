#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::iluvatar {

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t seqlen,
    size_t hidden_size,
    float eps,
    llaisysStream_t stream
);

} // namespace llaisys::ops::iluvatar