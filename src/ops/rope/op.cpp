#include "op.hpp"

#include <cmath>

#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/rope_iluvatar.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/rope_metax.hpp"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);

    CHECK_ARGUMENT(
        in->ndim() == 3,
        "RoPE: in must be a 3D tensor."
    );
    CHECK_ARGUMENT(
        out->ndim() == 3,
        "RoPE: out must be a 3D tensor."
    );
    CHECK_ARGUMENT(
        pos_ids->ndim() == 1,
        "RoPE: pos_ids must be a 1D tensor."
    );

    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(
        pos_ids->dtype() == LLAISYS_DTYPE_I64,
        "RoPE: pos_ids must have int64 dtype."
    );

    const size_t seq_len = in->shape()[0];
    const size_t n_heads = in->shape()[1];
    const size_t head_dim = in->shape()[2];

    CHECK_ARGUMENT(
        pos_ids->shape()[0] == seq_len,
        "RoPE: pos_ids length must equal sequence length."
    );

    CHECK_ARGUMENT(
        head_dim > 0 && head_dim % 2 == 0,
        "RoPE: head dimension must be positive and even."
    );

    CHECK_ARGUMENT(
        std::isfinite(theta) && theta > 0.0f,
        "RoPE: theta must be finite and positive."
    );

    ASSERT(
        out->isContiguous()
            && in->isContiguous()
            && pos_ids->isContiguous(),
        "RoPE: all tensors must be contiguous."
    );

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            theta,
            out->dtype(),
            seq_len,
            n_heads,
            head_dim
        );
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU: 
        return cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            theta,
            out->dtype(),
            seq_len,
            n_heads,
            head_dim
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            theta,
            out->dtype(),
            seq_len,
            n_heads,
            head_dim,
            llaisys::core::context().runtime().stream()
        );
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            theta,
            out->dtype(),
            seq_len,
            n_heads,
            head_dim,
            llaisys::core::context().runtime().stream()
        );
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            theta,
            out->dtype(),
            seq_len,
            n_heads,
            head_dim,
            llaisys::core::context().runtime().stream()
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
    
}
} // namespace llaisys::ops
