#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../cpu/cpu_utils.hpp"
#include "cpu/swiglu_cpu.hpp"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "../nvidia/nvidia_ops.cuh"
#endif
#ifdef ENABLE_ILUVATAR_API
    if (out->deviceType() == LLAISYS_DEVICE_ILUVATAR) {
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    }
#endif

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_ARGUMENT(out->ndim() == 2 && gate->ndim() == 2 && up->ndim() == 2,
                   "SwiGLU tensors must be 2D");
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_ARGUMENT(cpu::is_supported_float(out->dtype()), "SwiGLU only supports floating point tensors");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: all tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
