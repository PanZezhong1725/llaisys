#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"
#include "nvidia/swiglu_cuda.cuh"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->dtype() == gate->dtype() && out->dtype() == up->dtype(), "Swiglu: all tensors must have the same dtype.");
    ASSERT(gate->ndim() == 2 && up->ndim() == 2, "Swiglu: gate and up must be 2D.");
    ASSERT(out->ndim() == 2, "Swiglu: out must be 2D.");
    ASSERT(gate->shape()[0] == up->shape()[0] && gate->shape()[1] == up->shape()[1], "Swiglu: gate and up must have the same shape.");
    ASSERT(out->shape()[0] == gate->shape()[0] && out->shape()[1] == gate->shape()[1], "Swiglu: out must have the same shape as gate and up.");
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "Swiglu: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->shape()[0], out->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->shape()[0], out->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        cuda::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->shape()[0], out->shape()[1]);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
