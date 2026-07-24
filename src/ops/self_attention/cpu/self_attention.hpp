#pragma once

#include "llaisys.h"

#include <cstddef>

// V3(在 V2 基础上加了 GQA,不做 softmax 数值稳定处理):
// nhead 必须是 nkvhead 的整数倍,group = nhead / nkvhead 个相邻 query head 共享同一个 kv head:
// kvh(h) = h / group。
// causal: query 位置 i 只能看到 key 位置 j <= i + (total_len - seqlen)。
//
// attn_val : [seqlen,    nhead,   dv]
// q        : [seqlen,    nhead,   d ]
// k        : [total_len, nkvhead, d ]
// v        : [total_len, nkvhead, dv]
namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                     llaisysDataType_t type,
                     size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv,
                     float scale);
}
