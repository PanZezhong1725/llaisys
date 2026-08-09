#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(vals, max_idx, max_val);

    CHECK_ARGUMENT(vals->ndim() == 1,
                   "Argmax: vals must be one-dimensional.");
    CHECK_ARGUMENT(vals->numel() > 0,
                   "Argmax: vals cannot be empty.");
    CHECK_ARGUMENT(max_idx->numel() == 1,
                   "Argmax: max_idx must contain one element.");
    CHECK_ARGUMENT(max_val->numel() == 1,
                   "Argmax: max_val must contain one element.");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64,
                   "Argmax: max_idx must be int64.");
    CHECK_ARGUMENT(max_val->dtype() == vals->dtype(),
                   "Argmax: max_val dtype mismatch.");

    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "Argmax: all tensors must be contiguous.");

    // always support cpu calculation
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(),
                              vals->dtype(), vals->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
