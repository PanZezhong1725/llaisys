#include "op.hpp"
#include "cpu/rms_norm.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "nvidia/rms_norm_cuda.cuh"
#include "iluvatar/rms_norm_iluvatar.cuh"
//// RMSNorm:
// y[i] = w[i] * x[i] / sqrt(mean(x^2) + eps)
// where mean(x^2) = (1 / d) * sum_{j=0}^{d-1}(x[j] * x[j])
// w[i] is the weight for the i-th element, and eps is a small constant to avoid division by zero.
// in :[rows, d], weight :[d], out :[rows, d]
namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMSNorm: all tensors must be contiguous.");
    ASSERT(in->ndim() == 2, "RMSNorm: in must be 2D.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1D.");
    ASSERT(in->shape()[1] == weight->shape()[0], "RMSNorm: in's second dimension must match weight's first dimension.");
    ASSERT(out->shape()[0] == in->shape()[0], "RMSNorm: out's first dimension must match in's first dimension.");
    ASSERT(out->shape()[1] == in->shape()[1], "RMSNorm: out's second dimension must match in's second dimension.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        llaisys::ops::cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), in->shape()[0], in->shape()[1]);
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        llaisys::ops::cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), in->shape()[0], in->shape()[1]);
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        llaisys::ops::cuda::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), in->shape()[0], in->shape()[1]);
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        llaisys::ops::iluvatar::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), in->shape()[0], in->shape()[1]);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
