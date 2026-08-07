#include "op.hpp"
#include "./cpu/self_attention.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "nvidia/self_attention_cuda.cuh"
#include "iluvatar/self_attention_iluvatar.cuh"

// V3: 支持 GQA,要求 nhead 是 nkvhead 的整数倍。
// attn_val, q : [seqlen,    nhead,   d ] / [seqlen,    nhead,   dv]
// k, v        : [total_len, nkvhead, d ] / [total_len, nkvhead, dv]
namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && attn_val->ndim() == 3,
           "self_attention: q/k/v/attn_val must all be 3D.");
    ASSERT(q->shape()[1] % k->shape()[1] == 0,
           "self_attention (V3): nhead must be a multiple of nkvhead (GQA).");
    ASSERT(k->shape()[0] == v->shape()[0] && k->shape()[1] == v->shape()[1],
           "self_attention: k and v must share the same [total_len, nkvhead].");
    ASSERT(attn_val->shape()[0] == q->shape()[0] && attn_val->shape()[1] == q->shape()[1],
           "self_attention: attn_val's [seqlen, nhead] must match q's.");
    ASSERT(attn_val->shape()[2] == v->shape()[2],
           "self_attention: attn_val's dv must match v's dv.");

    size_t seqlen = q->shape()[0];
    size_t total_len = k->shape()[0];
    size_t nhead = q->shape()[1];
    size_t nkvhead = k->shape()[1];
    size_t d = q->shape()[2];
    size_t dv = v->shape()[2];

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        llaisys::ops::cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                           attn_val->dtype(), seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        llaisys::ops::cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                           attn_val->dtype(), seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        llaisys::ops::cuda::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                           attn_val->dtype(), seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        llaisys::ops::iluvatar::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                           attn_val->dtype(), seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
