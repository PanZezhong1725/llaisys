#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.hpp"
#endif

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: input, weight and out must be contiguous.");

    size_t seq_len = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    bool has_bias = bias->numel() > 0;

    if (has_bias) {
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(),
                           has_bias ? bias->data() : nullptr,
                           out->dtype(), seq_len, in_features, out_features, has_bias);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),
                           has_bias ? bias->data() : nullptr,
                           out->dtype(), seq_len, in_features, out_features, has_bias);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(),
                              has_bias ? bias->data() : nullptr,
                              out->dtype(), seq_len, in_features, out_features, has_bias);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops


