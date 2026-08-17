#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "icore/linear_nvidia.hpp"
#endif
#ifdef ENABLE_MUSA_API
#include "musa/linear_musa.hpp"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: all tensors must be contiguous.");
    // bias 可为空（无 bias 的线性层）
    if (bias) {
        CHECK_SAME_DEVICE(bias, out);
        CHECK_SAME_DTYPE(bias->dtype(), out->dtype());
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out, in, weight, bias, in->dtype(), in->numel());
    }

    llaisys::core::context().setDevice(in->deviceType(), out->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out, in, weight, bias, in->dtype(), in->numel());

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out, in, weight, bias, in->dtype(), in->numel());
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::linear(out, in, weight, bias, in->dtype(), in->numel());
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return musa::linear(out, in, weight, bias, in->dtype(), in->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
