#include "op.hpp"
#include "nvidia/rope_cuda.cuh"
#include "iluvatar/rope_iluvatar.cuh"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
// a' = a*cos(phi) - b*sin(phi)，b' = b*cos(phi) + a*sin(phi)
// out/in : [seqlen, nhead_or_nkvhead, d]；pos_ids : [seqlen,] int64；theta：频率向量基值。
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(in->ndim() == 3, "RoPE: in must be 3D.");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D.");
    ASSERT(out->shape()[0] == in->shape()[0], "RoPE: out's first dimension must match in's first dimension.");
    ASSERT(out->shape()[1] == in->shape()[1], "RoPE: out's second dimension must match in's second dimension.");

    // TODO: 没校验 pos_ids 的 dtype 恒为 int64
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), in->shape()[0], in->shape()[1], in->shape()[2], theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), in->shape()[0], in->shape()[1], in->shape()[2], theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        cuda::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), in->shape()[0], in->shape()[1], in->shape()[2], theta);
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        iluvatar::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), in->shape()[0], in->shape()[1], in->shape()[2], theta);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
