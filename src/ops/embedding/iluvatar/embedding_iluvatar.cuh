#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::iluvatar {

void embedding(
    std::byte *out,
    const std::byte *indices,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t seq_len,
    size_t vocab_size,
    size_t hidden_size,
    llaisysStream_t stream_
);

}