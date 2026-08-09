#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif
#ifdef ENABLE_COREX_API
#include "corex/self_attention_corex.cuh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3
                       && k->ndim() == 3 && v->ndim() == 3,
                   "SelfAttention: all tensors must be three-dimensional.");
    CHECK_ARGUMENT(q->shape()[0] > 0,
                   "SelfAttention: query length cannot be zero.");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0],
                   "SelfAttention: key and value lengths mismatch.");
    CHECK_ARGUMENT(k->shape()[0] >= q->shape()[0],
                   "SelfAttention: KV length cannot be shorter than query length.");
    CHECK_ARGUMENT(q->shape()[2] > 0 && q->shape()[2] == k->shape()[2],
                   "SelfAttention: query and key head dimensions mismatch.");
    CHECK_ARGUMENT(k->shape()[1] > 0 && k->shape()[1] == v->shape()[1],
                   "SelfAttention: key and value head counts mismatch.");
    CHECK_ARGUMENT(q->shape()[1] > 0 && q->shape()[1] % k->shape()[1] == 0,
                   "SelfAttention: query heads must be divisible by KV heads.");
    CHECK_ARGUMENT(v->shape()[2] > 0,
                   "SelfAttention: value head dimension cannot be zero.");
    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0]
                       && attn_val->shape()[1] == q->shape()[1]
                       && attn_val->shape()[2] == v->shape()[2],
                   "SelfAttention: output shape mismatch.");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(scale > 0.0f,
                   "SelfAttention: scale must be positive.");

    ASSERT(attn_val->isContiguous() && q->isContiguous()
               && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    const size_t query_len = q->shape()[0];
    const size_t kv_len = k->shape()[0];
    const size_t num_heads = q->shape()[1];
    const size_t num_kv_heads = k->shape()[1];
    const size_t head_dim = q->shape()[2];
    const size_t value_dim = v->shape()[2];

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(), scale, query_len, kv_len, num_heads,
            num_kv_heads, head_dim, value_dim);
    }

    llaisys::core::context().setDevice(
        attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(), scale, query_len, kv_len, num_heads,
            num_kv_heads, head_dim, value_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(), scale, query_len, kv_len, num_heads,
            num_kv_heads, head_dim, value_dim);
#endif
#ifdef ENABLE_COREX_API
    case LLAISYS_DEVICE_COREX:
        return corex::self_attention(
            attn_val->data(), q->data(), k->data(), v->data(),
            attn_val->dtype(), scale, query_len, kv_len, num_heads,
            num_kv_heads, head_dim, value_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
