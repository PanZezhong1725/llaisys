#include "op.hpp"

#include "cpu/swiglu_cpu.hpp"
namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);

    CHECK_ARGUMENT(
        out->ndim() == 2,
        "SwiGLU: out must be a 2D tensor."
    );

    CHECK_ARGUMENT(
        gate->ndim() == 2,
        "SwiGLU: gate must be a 2D tensor."
    );

    CHECK_ARGUMENT(
        up->ndim() == 2,
        "SwiGLU: up must be a 2D tensor."
    );

    CHECK_SAME_SHAPE(
        out->shape(),
        gate->shape(),
        up->shape()
    );

    CHECK_SAME_DTYPE(
        out->dtype(),
        gate->dtype(),
        up->dtype()
    );

    ASSERT(
        out->isContiguous()
            && gate->isContiguous()
            && up->isContiguous(),
        "SwiGLU: all tensors must be contiguous."
    );

    // 3个张量形状相同且连续，不需要分别遍历 seqlen 和 intermediate_size，直接遍历 numel() 即可
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            out->numel()
        );
    }

    llaisys::core::context().setDevice(
        out->deviceType(),
        out->deviceId()
    );

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            out->numel()
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
