#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../cpu/cpu_utils.hpp"
#include "cpu/rms_norm_cpu.hpp"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "../nvidia/nvidia_ops.cuh"
#endif
#ifdef ENABLE_ILUVATAR_API
    if (out->deviceType() == LLAISYS_DEVICE_ILUVATAR) {
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), in->dtype(), in->shape()[0], in->shape()[1], eps);
    }
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2, "RMS norm input and output must be 2D tensors");
    CHECK_ARGUMENT(weight->ndim() == 1, "RMS norm weight must be a 1D tensor");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(cpu::is_supported_float(in->dtype()), "RMS norm only supports floating point tensors");
    CHECK_ARGUMENT(in->shape()[1] > 0 && weight->shape()[0] == in->shape()[1],
                   "RMS norm weight must match the input row width");
    CHECK_ARGUMENT(eps >= 0.0F, "RMS norm epsilon must be non-negative");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMS norm: all tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), in->dtype(), in->shape()[0], in->shape()[1], eps);
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), in->dtype(), in->shape()[0], in->shape()[1], eps);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
