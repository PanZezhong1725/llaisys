#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"
#include "../cpu/cpu_utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/nvidia_ops.cuh"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() == 1 && vals->numel() > 0, "Argmax input must be a non-empty 1D tensor");
    CHECK_ARGUMENT(max_idx->ndim() == 1 && max_idx->shape()[0] == 1, "Argmax index output must have shape [1]");
    CHECK_ARGUMENT(max_val->ndim() == 1 && max_val->shape()[0] == 1, "Argmax value output must have shape [1]");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax index output must use int64");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(cpu::is_supported_float(vals->dtype()), "Argmax only supports floating point input");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "Argmax: all tensors must be contiguous.");
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (vals->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
