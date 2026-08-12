#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../cpu/cpu_utils.hpp"
#include "cpu/self_attention_cpu.hpp"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "../nvidia/nvidia_ops.cuh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                   "Self-attention tensors must be 3D");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(cpu::is_supported_float(q->dtype()), "Self-attention only supports floating point tensors");
    const size_t query_length = q->shape()[0];
    const size_t key_length = k->shape()[0];
    const size_t query_heads = q->shape()[1];
    const size_t key_value_heads = k->shape()[1];
    const size_t head_dim = q->shape()[2];
    const size_t value_dim = v->shape()[2];
    CHECK_ARGUMENT(query_length > 0 && key_length >= query_length, "Self-attention requires key length >= query length");
    CHECK_ARGUMENT(query_heads > 0 && key_value_heads > 0 && query_heads % key_value_heads == 0,
                   "Query heads must be divisible by key/value heads");
    CHECK_ARGUMENT(k->shape()[1] == v->shape()[1] && k->shape()[2] == head_dim,
                   "Key and value shapes are incompatible with query");
    CHECK_ARGUMENT(attn_val->shape()[0] == query_length && attn_val->shape()[1] == query_heads
                       && attn_val->shape()[2] == value_dim,
                   "Self-attention output shape is incompatible with inputs");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "Self-attention: all tensors must be contiguous.");
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q->dtype(), query_length,
                                   key_length, query_heads, key_value_heads, head_dim, value_dim, scale);
    }
    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q->dtype(), query_length,
                                      key_length, query_heads, key_value_heads, head_dim, value_dim, scale);
    }
#endif
#ifdef ENABLE_ILUVATAR_API
    if (attn_val->deviceType() == LLAISYS_DEVICE_ILUVATAR) {
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q->dtype(), query_length,
                                      key_length, query_heads, key_value_heads, head_dim, value_dim, scale);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
