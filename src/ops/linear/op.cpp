#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../cpu/cpu_utils.hpp"
#include "cpu/linear_cpu.hpp"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "../nvidia/nvidia_ops.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
                   "Linear input, weight and output must be 2D tensors");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(cpu::is_supported_float(out->dtype()), "Linear only supports floating point tensors");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "Linear input and weight dimensions do not match");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0],
                   "Linear output shape is incompatible with input and weight");
    if (bias != nullptr) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == weight->shape()[0],
                       "Linear bias shape is incompatible with weight");
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous()
               && (bias == nullptr || bias->isContiguous()),
           "Linear: all tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias == nullptr ? nullptr : bias->data(), out->dtype(),
                           in->shape()[0], in->shape()[1], weight->shape()[0]);
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::linear(out->data(), in->data(), weight->data(), bias == nullptr ? nullptr : bias->data(),
                              out->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
    }
#endif
#ifdef ENABLE_ILUVATAR_API
    if (out->deviceType() == LLAISYS_DEVICE_ILUVATAR) {
        return nvidia::linear(out->data(), in->data(), weight->data(), bias == nullptr ? nullptr : bias->data(),
                              out->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
