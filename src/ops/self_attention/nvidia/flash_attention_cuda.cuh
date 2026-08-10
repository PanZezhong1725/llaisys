#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {

void flash_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                     llaisysDataType_t type,
                     size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv,
                     float scale);

// seqlen=1 的 decode 专用入口，对完整 KV cache 执行 attention。
void flash_attention_decode(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                            llaisysDataType_t type,
                            size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv,
                            float scale);

// seqlen=1 且 total_len 较大时的 decode 入口：把 KV 方向切成多段并行规约（flash-decoding），
// 缓解 flash_attention_decode 只用 nhead 个 warp、SM 打不满的问题。
void flash_attention_decode_splitkv(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                                    llaisysDataType_t type,
                                    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv,
                                    float scale);

}
