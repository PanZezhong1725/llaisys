#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);

    // 弃用：永久 stub，不会实现（KV cache 拷贝等场景改用 memcpy，见 CLAUDE.md）。
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        TO_BE_IMPLEMENTED();
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        TO_BE_IMPLEMENTED();
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
