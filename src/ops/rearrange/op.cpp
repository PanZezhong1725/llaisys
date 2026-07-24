#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);

    // TODO: rearrange 特有的校验(dtype/shape 等)

    // TODO: 设计 cpu kernel 接口后,#include "cpu/rearrange_cpu.hpp" 并把下面的 TO_BE_IMPLEMENTED() 换成实际调用

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
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
