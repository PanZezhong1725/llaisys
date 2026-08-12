#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../cpu/cpu_utils.hpp"
#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/nvidia_ops.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_ARGUMENT(out->ndim() == 3 && in->ndim() == 3, "RoPE input and output must be 3D tensors");
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->dtype() == LLAISYS_DTYPE_I64,
                   "RoPE position ids must be a 1D int64 tensor");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(cpu::is_supported_float(in->dtype()), "RoPE only supports floating point tensors");
    CHECK_ARGUMENT(in->shape()[2] > 0 && in->shape()[2] % 2 == 0, "RoPE head dimension must be positive and even");
    CHECK_ARGUMENT(pos_ids->shape()[0] == in->shape()[0], "RoPE position count must match sequence length");
    CHECK_ARGUMENT(theta > 0.0F, "RoPE theta must be positive");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), in->dtype(), in->shape()[0], in->shape()[1],
                         in->shape()[2], theta);
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), in->dtype(), in->shape()[0], in->shape()[1],
                            in->shape()[2], theta);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
