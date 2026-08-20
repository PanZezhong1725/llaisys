#include "op.hpp"

#include <cmath>

#include "cpu/self_attention_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/self_attention_iluvatar.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/self_attention_metax.hpp"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    CHECK_ARGUMENT(
        attn_val->ndim() == 3,
        "SelfAttention: attn_val must be a 3D tensor."
    );

    CHECK_ARGUMENT(
        q->ndim() == 3,
        "SelfAttention: q must be a 3D tensor."
    );

    CHECK_ARGUMENT(
        k->ndim() == 3,
        "SelfAttention: k must be a 3D tensor."
    );

    CHECK_ARGUMENT(
        v->ndim() == 3,
        "SelfAttention: v must be a 3D tensor."
    );

    ASSERT(
        attn_val->isContiguous()
            && q->isContiguous()
            && k->isContiguous()
            && v->isContiguous(),
        "SelfAttention: all tensors must be contiguous."
    );

    const size_t q_len = q->shape()[0];       // 当前需要计算注意力的 token 数
    const size_t q_heads = q->shape()[1];     // Query 头的数量
    const size_t qk_dim = q->shape()[2];      // 每个 Query/Key head 的特征维度
  
    const size_t total_len = k->shape()[0];   // 所有可被查询的 token 数，包括历史 KV Cache
    const size_t kv_heads = k->shape()[1];    // Key / Value 的头数

    const size_t v_total_len = v->shape()[0]; // 所有可被查询的 token 数，包括历史 KV Cache
    const size_t v_kv_heads = v->shape()[1];  //  Key / Value 的头数
    const size_t v_dim = v->shape()[2];       // 每个 Value 向量的长度


    CHECK_ARGUMENT(
        q_len > 0,
        "SelfAttention: q sequence length must be positive."
    );

    CHECK_ARGUMENT(
        total_len >= q_len,
        "SelfAttention: total KV length must be at least the query sequence length."
    );

    CHECK_ARGUMENT(
        q_heads > 0 && kv_heads > 0,
        "SelfAttention: head counts must be positive."
    );

    CHECK_ARGUMENT(
        qk_dim > 0 && v_dim > 0,
        "SelfAttention: head dimensions must be positive."
    );

    CHECK_ARGUMENT(
        k->shape()[2] == qk_dim,
        "SelfAttention: q and k must have the same "
        "head dimension."
    );

    CHECK_ARGUMENT(
        v_total_len == total_len,
        "SelfAttention: k and v must have the same "
        "sequence length."
    );

    CHECK_ARGUMENT(
        v_kv_heads == kv_heads,
        "SelfAttention: k and v must have the same "
        "number of KV heads."
    );

    CHECK_ARGUMENT(
        q_heads % kv_heads == 0,
        "SelfAttention: the number of query heads must be divisible by the number of KV heads."
    );

    CHECK_ARGUMENT(
        attn_val->shape()[0] == q_len
            && attn_val->shape()[1] == q_heads
            && attn_val->shape()[2] == v_dim,
        "SelfAttention: attn_val shape must be "
        "[q_len, q_heads, v_dim]."
    );

    CHECK_ARGUMENT(
        std::isfinite(scale),
        "SelfAttention: scale must be finite."
    );

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            scale,
            attn_val->dtype(),
            q_len,
            total_len,
            q_heads,
            kv_heads,
            qk_dim,
            v_dim
        );
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            scale,
            attn_val->dtype(),
            q_len,
            total_len,
            q_heads,
            kv_heads,
            qk_dim,
            v_dim
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            scale,
            q_len,
            total_len,
            q_heads,
            kv_heads,
            qk_dim,
            v_dim,
            llaisys::core::context().runtime().stream()
        );
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            scale,
            q_len,
            total_len,
            q_heads,
            kv_heads,
            qk_dim,
            v_dim,
            llaisys::core::context().runtime().stream()
        );
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            scale,
            q_len,
            total_len,
            q_heads,
            kv_heads,
            qk_dim,
            v_dim,
            llaisys::core::context().runtime().stream()
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
