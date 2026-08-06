#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "SelfAttention: all tensors must be contiguous.");
    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3D.");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3D.");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3D.");
    ASSERT(attn_val->ndim() == 3, "SelfAttention: output must be 3D.");
    ASSERT(q->shape()[0] == attn_val->shape()[0] && q->shape()[1] == attn_val->shape()[1] && q->shape()[2] == attn_val->shape()[2], "SelfAttention: output shape mismatch.");
    ASSERT(k->shape()[1] == v->shape()[1], "SelfAttention: k and v head dimension mismatch.");
    ASSERT(q->shape()[2] == k->shape()[2], "SelfAttention: q and k head dimension mismatch.");
    ASSERT(k->shape()[0] == v->shape()[0], "SelfAttention: k and v sequence length mismatch.");

    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];
    size_t hd = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), qlen, kvlen, nh, nkvh, hd);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), qlen, kvlen, nh, nkvh, hd);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
