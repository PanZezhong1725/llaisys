#include "op.hpp"
#include "nvidia/rope_cuda.cuh"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
// a[i][j]' = a[i][j] * cos(phi[i][j]) - b[i][j] * sin(phi[i][j])
// b[i][j]' = b[i][j] * cos(phi[i][j]) + a[i][j] * sin(phi[i][j])
    // a' = a * cos(phi) - b * sin(phi)
    // b' = b * cos(phi) + a * sin(phi)
    // RoPE (Rotary Position Embedding):
//     out：结果q或k张量。形状应该是 [seqlen, nhead, d] 或 [seqlen, nkvhead, d]。你暂时可以假设张量是连续的。
// in：原始q或k张量。形状应该是 [seqlen, nhead, d] 或 [seqlen, nkvhead, d]。你暂时可以假设张量是连续的。
// pos_ids：输入序列中每个token的位置id（整个上下文中的索引）。形状应该是 [seqlen,]，dtype应该是int64。
// theta：频率向量的基值。
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(in->ndim() == 3, "RoPE: in must be 3D.");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D.");
    ASSERT(out->shape()[0] == in->shape()[0], "RoPE: out's first dimension must match in's first dimension.");
    ASSERT(out->shape()[1] == in->shape()[1], "RoPE: out's second dimension must match in's second dimension.");
    
    // RoPE (Rotary Position Embedding):

    // TODO: pos_ids 的 dtype 通常和 in/out 不同(应恒为 int64),这里还没校验

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
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
