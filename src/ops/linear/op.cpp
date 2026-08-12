#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"
#include "llaisys.h"
#include <cstddef>

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/linear_iluvatar.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/linear_metax.hpp"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);

    if (bias) {
        CHECK_SAME_DEVICE(weight, bias);
    }

    // 当前假设是 2D 连续张量
    CHECK_ARGUMENT(
        out->ndim() == 2,
        "Linear: out must be a 2D tensor."
    );
    CHECK_ARGUMENT(
        in->ndim() == 2,
        "Linear: in must be a 2D tensor."
    );
    CHECK_ARGUMENT(
        weight->ndim() == 2,
        "Linear: weight must be a 2D tensor."
    );
    if (bias) {
        CHECK_ARGUMENT(
            bias->ndim() == 1,
            "Linear: bias must be a 1D tensor."
        );
    }

    // in:     [M, K]
    // weight: [N, K]
    // bias:   [N]
    // out:    [M, N]
    const size_t m = in->shape()[0];
    const size_t k = in->shape()[1];
    const size_t n = weight->shape()[0];

    // 检查能否矩阵乘
    CHECK_ARGUMENT(
        k == weight->shape()[1],
        "Linear: in.shape[1] must equal weight.shape[1]."
    );
    CHECK_ARGUMENT(
        m == out->shape()[0] && n == out->shape()[1],
        "Linear: out shape must be [in.shape[0], weight.shape[0]]."
    );
    if (bias) {
        CHECK_ARGUMENT(
            n == bias->shape()[0],
            "Linear: bias.shape[0] must equal weight.shape[0]."
        );
    }

    ASSERT(
        out->isContiguous() &&
        in->isContiguous() &&
        weight->isContiguous() &&
        (!bias || bias->isContiguous()),
        "Linear: all tensors must be contiguous."
    );

    // 防止 nullptr->data()
    const std::byte *bias_data = bias ? bias->data() : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            m,
            n,
            k
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU: 
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            m,
            n,
            k
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            m,
            n,
            k,
            llaisys::core::context().runtime().stream()
        );
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            m,
            n,
            k,
            llaisys::core::context().runtime().stream()
        );
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            m,
            n,
            k,
            llaisys::core::context().runtime().stream()
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
