#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "icore/self_attention_nvidia.hpp"
#endif
#ifdef ENABLE_MUSA_API
#include "musa/self_attention_musa.hpp"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val, q, k, v, scale, attn_val->dtype());
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()){
        case LLAISYS_DEVICE_CPU:
            return cpu::self_attention(attn_val, q, k, v, scale, attn_val->dtype());

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val, q, k, v, scale, attn_val->dtype());
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::self_attention(attn_val, q, k, v, scale, attn_val->dtype());
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return musa::self_attention(attn_val, q, k, v, scale, attn_val->dtype());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
