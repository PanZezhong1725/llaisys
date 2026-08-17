#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "icore/swiglu_nvidia.hpp"
#endif
#ifdef ENABLE_MUSA_API
#include "musa/swiglu_musa.hpp"
#endif

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: all tensors must be contiguous.");

    if (up->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out, gate, up, gate->dtype(), gate->numel());
    }

    llaisys::core::context().setDevice(gate->deviceType(), gate->deviceId());

    switch (up->deviceType()){
        case LLAISYS_DEVICE_CPU:
            return cpu::swiglu(out, gate, up, gate->dtype(), gate->numel());

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(out, gate, up, gate->dtype(), gate->numel());
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::swiglu(out, gate, up, gate->dtype(), gate->numel());
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return musa::swiglu(out, gate, up, gate->dtype(), gate->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
