#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"
#endif

namespace llaisys::ops {

void swiglu(tensor_t output, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(output, gate, up);
    CHECK_SAME_SHAPE(output->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(output->dtype(), gate->dtype(), up->dtype());
    CHECK_ARGUMENT(output->isContiguous() && gate->isContiguous() && up->isContiguous(), "swiglu requires contiguous tensors");

    core::context().setDevice(output->deviceType(), output->deviceId());
    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(output->data(), gate->data(), up->data(), output->dtype(), output->numel());
    }
#ifdef ENABLE_NVIDIA_API
    if (output->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::swiglu(output->data(), gate->data(), up->data(), output->dtype(), output->numel(), core::context().runtime().stream());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
