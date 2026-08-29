#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: all tensors must be contiguous.");
    ASSERT(in->ndim() == 2, "Linear: input must be 2D.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D.");
    ASSERT(out->ndim() == 2, "Linear: output must be 2D.");
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: input and weight feature dimension mismatch.");
    ASSERT(out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0], "Linear: output shape mismatch.");
    if (bias != nullptr) {
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == weight->shape()[0], "Linear: bias shape mismatch.");
    }

    size_t batch = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), batch, in_features, out_features);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), batch, in_features, out_features);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), batch, in_features, out_features, out->deviceId());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
