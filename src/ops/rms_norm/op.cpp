#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.hpp"
#endif
#ifdef ENABLE_SUDA_API
#include "suda/rms_norm_suda.hpp"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RmsNorm: all tensors must be contiguous.");

    size_t seq_len = in->shape()[0];
    size_t hidden_size = in->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             out->dtype(), seq_len, hidden_size, eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(),
                             out->dtype(), seq_len, hidden_size, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(),
                                out->dtype(), seq_len, hidden_size, eps);
#endif
#ifdef ENABLE_SUDA_API
    case LLAISYS_DEVICE_SUDA:
        return suda::rms_norm(out->data(), in->data(), weight->data(),
                              out->dtype(), seq_len, hidden_size, eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops


