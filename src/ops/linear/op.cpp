#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "nvidia/linear_cuda.cuh"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(in->ndim() == 2, "Linear: in must be 2D.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D.");
    ASSERT(in->shape()[1] == weight->shape()[1], "Linear: in's second dimension must match weight's second dimension.");
    ASSERT(out->shape()[0] == in->shape()[0], "Linear: out's first dimension must match in's first dimension.");
    ASSERT(out->shape()[1] == weight->shape()[0], "Linear: out's second dimension must match weight's first dimension.");

    std::byte *bias_ptr = nullptr;
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == weight->shape()[0],
               "Linear: bias must be 1D with size equal to weight's first dimension.");
        bias_ptr = bias->data();
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_ptr,
                            out->dtype(), in->shape()[0], weight->shape()[0], in->shape()[1]);
    #ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA: {
            auto resource = llaisys::core::context().runtime().resource();
            cuda::linear(out->data(), in->data(), weight->data(), bias_ptr,
                            out->dtype(), in->shape()[0], weight->shape()[0], in->shape()[1],resource);
            return;
        }
    #endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
