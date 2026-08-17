#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/add_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/add_nvidia.cuh"
#endif

namespace llaisys::ops {

void add(tensor_t output, tensor_t left, tensor_t right) {
    CHECK_SAME_DEVICE(output, left, right);
    CHECK_SAME_SHAPE(output->shape(), left->shape(), right->shape());
    CHECK_SAME_DTYPE(output->dtype(), left->dtype(), right->dtype());
    CHECK_ARGUMENT(output->isContiguous() && left->isContiguous() && right->isContiguous(), "add requires contiguous tensors");

    core::context().setDevice(output->deviceType(), output->deviceId());
    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::add(output->data(), left->data(), right->data(), output->dtype(), output->numel());
    }
#ifdef ENABLE_NVIDIA_API
    if (output->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::add(output->data(), left->data(), right->data(), output->dtype(), output->numel(), core::context().runtime().stream());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
