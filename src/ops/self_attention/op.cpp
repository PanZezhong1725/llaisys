#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cstring>
#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.hpp"
#endif
#ifdef ENABLE_SUDA_API
#include "suda/self_attention_suda.hpp"
#endif

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    size_t seq_len = q->shape()[0];
    size_t num_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];
    size_t num_kv_heads = k->shape()[1];
    size_t kv_len = k->shape()[0];

    // Handle GQA: repeat K/V heads to match Q heads
    // If num_heads == num_kv_heads, no repetition needed
    // Otherwise, repeat each KV head (num_heads / num_kv_heads) times
    size_t num_repeats = num_heads / num_kv_heads;

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        if (num_repeats > 1) {
            // Need to expand K and V by repeating heads
            size_t elem_size = 0;
            switch (q->dtype()) {
            case LLAISYS_DTYPE_F32: elem_size = 4; break;
            case LLAISYS_DTYPE_F16:
            case LLAISYS_DTYPE_BF16: elem_size = 2; break;
            default: return;
            }

            // Create expanded K and V buffers (kv_len, not seq_len, since K/V may have different length)
            // Original K/V layout: [kv_len, num_kv_heads, head_dim]
            // Expanded K/V layout: [kv_len, num_heads, head_dim]
            // Element (i, h, k) in expanded is at (i * num_heads * head_dim + h * head_dim + k) * elem_size
            // Element (i, h, k) in original is at (i * num_kv_heads * head_dim + h * head_dim + k) * elem_size
            std::vector<std::byte> k_expanded(kv_len * num_heads * head_dim * elem_size);
            std::vector<std::byte> v_expanded(kv_len * num_heads * head_dim * elem_size);

            size_t kv_stride = num_kv_heads * head_dim;
            size_t expanded_stride = num_heads * head_dim;

            for (size_t h = 0; h < num_heads; h++) {
                size_t src_kv_head = h / num_repeats;
                for (size_t i = 0; i < kv_len; i++) {
                    for (size_t kd = 0; kd < head_dim; kd++) {
                        size_t src_byte_offset = (i * kv_stride + src_kv_head * head_dim + kd) * elem_size;
                        size_t dst_byte_offset = (i * expanded_stride + h * head_dim + kd) * elem_size;
                        std::memcpy(k_expanded.data() + dst_byte_offset,
                                   reinterpret_cast<const std::byte *>(k->data()) + src_byte_offset,
                                   elem_size);
                        std::memcpy(v_expanded.data() + dst_byte_offset,
                                   reinterpret_cast<const std::byte *>(v->data()) + src_byte_offset,
                                   elem_size);
                    }
                }
            }

            return cpu::self_attention(attn_val->data(), q->data(),
                                       k_expanded.data(), v_expanded.data(),
                                       attn_val->dtype(), seq_len, kv_len, num_heads, head_dim, scale);
        }

        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), seq_len, kv_len, num_heads, head_dim, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        if (num_repeats > 1) {
            size_t elem_size = 0;
            switch (q->dtype()) {
            case LLAISYS_DTYPE_F32: elem_size = 4; break;
            case LLAISYS_DTYPE_F16:
            case LLAISYS_DTYPE_BF16: elem_size = 2; break;
            default: return;
            }

            size_t kv_stride = num_kv_heads * head_dim;
            size_t expanded_stride = num_heads * head_dim;
            std::vector<std::byte> k_expanded(kv_len * num_heads * head_dim * elem_size);
            std::vector<std::byte> v_expanded(kv_len * num_heads * head_dim * elem_size);

            for (size_t h = 0; h < num_heads; h++) {
                size_t src_kv_head = h / num_repeats;
                for (size_t i = 0; i < kv_len; i++) {
                    for (size_t kd = 0; kd < head_dim; kd++) {
                        size_t src_byte_offset = (i * kv_stride + src_kv_head * head_dim + kd) * elem_size;
                        size_t dst_byte_offset = (i * expanded_stride + h * head_dim + kd) * elem_size;
                        std::memcpy(k_expanded.data() + dst_byte_offset,
                                   reinterpret_cast<const std::byte *>(k->data()) + src_byte_offset,
                                   elem_size);
                        std::memcpy(v_expanded.data() + dst_byte_offset,
                                   reinterpret_cast<const std::byte *>(v->data()) + src_byte_offset,
                                   elem_size);
                    }
                }
            }

            return cpu::self_attention(attn_val->data(), q->data(),
                                       k_expanded.data(), v_expanded.data(),
                                       attn_val->dtype(), seq_len, kv_len, num_heads, head_dim, scale);
        }
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), seq_len, kv_len, num_heads, head_dim, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                      attn_val->dtype(), seq_len, kv_len, num_heads,
                                      num_kv_heads, head_dim, scale);
#endif
#ifdef ENABLE_SUDA_API
    case LLAISYS_DEVICE_SUDA:
        return suda::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                    attn_val->dtype(), seq_len, kv_len, num_heads,
                                    num_kv_heads, head_dim, scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops


