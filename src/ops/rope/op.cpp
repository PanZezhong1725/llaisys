#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_ARGUMENT(out->ndim() == 3,
                   "RoPE: out must be three-dimensional.");
    CHECK_ARGUMENT(in->ndim() == 3,
                   "RoPE: input must be three-dimensional.");
    CHECK_ARGUMENT(pos_ids->ndim() == 1,
                   "RoPE: pos_ids must be one-dimensional.");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64,
                   "RoPE: pos_ids must be int64.");
    CHECK_ARGUMENT(pos_ids->shape()[0] == in->shape()[0],
                   "RoPE: pos_ids length must equal the sequence length.");
    CHECK_ARGUMENT(in->shape()[2] > 0 && in->shape()[2] % 2 == 0,
                   "RoPE: head dimension must be a non-zero even number.");
    CHECK_ARGUMENT(theta > 0.0f,
                   "RoPE: theta must be positive.");

    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");

    const size_t seq_len = in->shape()[0];
    const size_t num_heads = in->shape()[1];
    const size_t head_dim = in->shape()[2];

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(), in->data(), pos_ids->data(), out->dtype(), theta,
            seq_len, num_heads, head_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(
            out->data(), in->data(), pos_ids->data(), out->dtype(), theta,
            seq_len, num_heads, head_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(
            out->data(), in->data(), pos_ids->data(), out->dtype(), theta,
            seq_len, num_heads, head_dim);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
