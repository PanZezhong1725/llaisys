#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMSNorm: all tensors must be contiguous.");
    ASSERT(in->ndim() == 2, "RMSNorm: input must be 2D.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1D.");
    ASSERT(out->ndim() == 2, "RMSNorm: output must be 2D.");
    ASSERT(in->shape()[1] == weight->shape()[0], "RMSNorm: input and weight dimension mismatch.");
    ASSERT(out->shape()[0] == in->shape()[0] && out->shape()[1] == in->shape()[1], "RMSNorm: output shape mismatch.");

    size_t rows = in->shape()[0];
    size_t cols = in->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), rows, cols);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), rows, cols);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), rows, cols);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
