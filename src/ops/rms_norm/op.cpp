#include "op.hpp"
#include <cstddef>

#include "cpu/rms_norm_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(
        out->dtype(),
        in->dtype(),
        weight->dtype()
    );

    CHECK_ARGUMENT(
        in->ndim() >= 1,
        "RMSNorm: input must have at least one dimension."
    );

    CHECK_ARGUMENT(
        weight->ndim() == 1,
        "RMSNorm: weight must be a 1D tensor."
    );

    const size_t norm_size = in->shape().back();

    CHECK_ARGUMENT(
        norm_size > 0,
        "RMSNorm: normalized dimension must not be empty."
    );

    CHECK_ARGUMENT(
        weight->numel() == norm_size,
        "RMSNorm: weight size must equal "
        "the last dimension of input."
    );

    CHECK_ARGUMENT(
        eps >= 0.0f,
        "RMSNorm: eps must be non-negative."
    );

    ASSERT(
        out->isContiguous() &&
        in->isContiguous() &&
        weight->isContiguous(),
        "RMSNorm: all tensors must be contiguous."
    );

    // 归一化分组数量
    const size_t num_groups = in->numel() / norm_size;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            eps,
            out->dtype(),
            num_groups,
            norm_size
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU: 
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            eps,
            out->dtype(),
            num_groups,
            norm_size
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            num_groups,
            norm_size,
            eps,
            llaisys::core::context().runtime().stream()
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
