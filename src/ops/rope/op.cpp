#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

namespace llaisys::ops {

void rope(tensor_t output, tensor_t input, tensor_t positions, float theta) {
    CHECK_SAME_DEVICE(output, input, positions);
    CHECK_SAME_SHAPE(output->shape(), input->shape());
    CHECK_ARGUMENT(input->ndim() == 3 && positions->ndim() == 1, "rope expects [S,H,D] and [S]");
    CHECK_ARGUMENT(positions->shape()[0] == input->shape()[0], "rope position count mismatch");
    CHECK_ARGUMENT(positions->dtype() == LLAISYS_DTYPE_I64, "rope positions must be int64");
    CHECK_ARGUMENT(output->dtype() == input->dtype() && input->shape()[2] % 2 == 0, "rope dtype or head width is invalid");
    CHECK_ARGUMENT(output->isContiguous() && input->isContiguous() && positions->isContiguous(), "rope requires contiguous tensors");

    core::context().setDevice(output->deviceType(), output->deviceId());
    const size_t sequence = input->shape()[0];
    const size_t heads = input->shape()[1];
    const size_t width = input->shape()[2];
    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(output->data(), input->data(), positions->data(), output->dtype(), sequence, heads, width, theta);
    }
#ifdef ENABLE_NVIDIA_API
    if (output->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::rope(output->data(), input->data(), positions->data(), output->dtype(), sequence, heads, width, theta, core::context().runtime().stream());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
