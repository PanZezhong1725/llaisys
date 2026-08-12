#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "icore/rms_norm_nvidia.hpp"
#endif
#ifdef ENABLE_MUSA_API
#include "musa/rms_norm_musa.hpp"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out, in, weight, out->dtype(), eps);
    }

    llaisys::core::context().setDevice(in->deviceType(), out->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out, in, weight, out->dtype(), eps);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out, in, weight, out->dtype(), eps);
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::rms_norm(out, in, weight, out->dtype(), eps);
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return musa::rms_norm(out, in, weight, out->dtype(), eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
