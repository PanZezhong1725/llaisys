#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }

    CHECK_ARGUMENT(out->ndim() == 2,
                   "Linear: out must be two-dimensional.");
    CHECK_ARGUMENT(in->ndim() == 2,
                   "Linear: input must be two-dimensional.");
    CHECK_ARGUMENT(weight->ndim() == 2,
                   "Linear: weight must be two-dimensional.");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1],
                   "Linear: input and weight feature dimensions mismatch.");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0],
                   "Linear: output row count mismatch.");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0],
                   "Linear: output feature count mismatch.");
    CHECK_ARGUMENT(out->dtype() == in->dtype() && in->dtype() == weight->dtype(),
                   "Linear: out, input and weight dtype mismatch.");

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: out, input and weight must be contiguous.");

    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->ndim() == 1,
                       "Linear: bias must be one-dimensional.");
        CHECK_ARGUMENT(bias->shape()[0] == weight->shape()[0],
                       "Linear: bias size mismatch.");
        CHECK_ARGUMENT(bias->dtype() == in->dtype(),
                       "Linear: bias dtype mismatch.");
        ASSERT(bias->isContiguous(),
               "Linear: bias must be contiguous.");
    }

    const size_t m = in->shape()[0];
    const size_t n = weight->shape()[0];
    const size_t k = in->shape()[1];
    const std::byte *bias_data = bias == nullptr ? nullptr : bias->data();

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(), in->data(), weight->data(), bias_data,
            out->dtype(), m, n, k);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(
            out->data(), in->data(), weight->data(), bias_data,
            out->dtype(), m, n, k);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
