#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

namespace llaisys::ops {

void linear(tensor_t output, tensor_t input, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(output, input, weight);
    CHECK_ARGUMENT(output->ndim() == 2 && input->ndim() == 2 && weight->ndim() == 2, "linear expects rank-two tensors");
    CHECK_ARGUMENT(input->shape()[1] == weight->shape()[1], "linear reduction dimension mismatch");
    CHECK_ARGUMENT(output->shape()[0] == input->shape()[0] && output->shape()[1] == weight->shape()[0], "linear output shape mismatch");
    CHECK_SAME_DTYPE(output->dtype(), input->dtype(), weight->dtype());
    CHECK_ARGUMENT(output->isContiguous() && input->isContiguous() && weight->isContiguous(), "linear requires contiguous tensors");

    if (bias != nullptr) {
        CHECK_SAME_DEVICE(output, bias);
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == output->shape()[1], "linear bias shape mismatch");
        CHECK_ARGUMENT(bias->dtype() == output->dtype() && bias->isContiguous(), "linear bias metadata mismatch");
    }

    core::context().setDevice(output->deviceType(), output->deviceId());
    const size_t rows = input->shape()[0];
    const size_t columns = weight->shape()[0];
    const size_t reduction = input->shape()[1];
    const std::byte *bias_data = bias == nullptr ? nullptr : bias->data();
    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(output->data(), input->data(), weight->data(), bias_data, output->dtype(), rows, columns, reduction);
    }
#ifdef ENABLE_NVIDIA_API
    if (output->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::linear(output->data(), input->data(), weight->data(), bias_data, output->dtype(), rows, columns, reduction, core::context().runtime().stream());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
