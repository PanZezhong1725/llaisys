#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif

namespace llaisys::ops {

void argmax(tensor_t max_index, tensor_t max_value, tensor_t input) {
    CHECK_SAME_DEVICE(max_index, max_value, input);
    CHECK_ARGUMENT(max_index->numel() == 1 && max_value->numel() == 1, "argmax outputs must contain one element");
    CHECK_ARGUMENT(max_index->dtype() == LLAISYS_DTYPE_I64, "argmax index must be int64");
    CHECK_ARGUMENT(max_value->dtype() == input->dtype(), "argmax value dtype mismatch");
    CHECK_ARGUMENT(max_index->isContiguous() && max_value->isContiguous() && input->isContiguous(), "argmax requires contiguous tensors");

    core::context().setDevice(input->deviceType(), input->deviceId());
    if (input->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_index->data(), max_value->data(), input->data(), input->dtype(), input->numel());
    }
#ifdef ENABLE_NVIDIA_API
    if (input->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::argmax(max_index->data(), max_value->data(), input->data(), input->dtype(), input->numel(), core::context().runtime().stream());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
