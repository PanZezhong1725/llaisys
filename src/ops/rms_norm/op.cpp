#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif

namespace llaisys::ops {

void rms_norm(tensor_t output, tensor_t input, tensor_t weight, float epsilon) {
    CHECK_SAME_DEVICE(output, input, weight);
    CHECK_SAME_SHAPE(output->shape(), input->shape());
    CHECK_SAME_DTYPE(output->dtype(), input->dtype(), weight->dtype());
    CHECK_ARGUMENT(input->ndim() == 2 && weight->ndim() == 1, "rms_norm expects [M,N] and [N]");
    CHECK_ARGUMENT(weight->shape()[0] == input->shape()[1], "rms_norm weight shape mismatch");
    CHECK_ARGUMENT(output->isContiguous() && input->isContiguous() && weight->isContiguous(), "rms_norm requires contiguous tensors");

    core::context().setDevice(output->deviceType(), output->deviceId());
    const size_t rows = input->shape()[0];
    const size_t width = input->shape()[1];
    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(output->data(), input->data(), weight->data(), output->dtype(), rows, width, epsilon);
    }
#ifdef ENABLE_NVIDIA_API
    if (output->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::rms_norm(output->data(), input->data(), weight->data(), output->dtype(), rows, width, epsilon, core::context().runtime().stream());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
