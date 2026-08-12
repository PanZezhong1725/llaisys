#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);

    // 当前实现约定：
    //   index  是 1-D
    //   weight 是 2-D
    //   out    是 2-D
    CHECK_ARGUMENT(
        index->ndim() == 1,
        "Embedding: index must be a 1D tensor."
    );

    CHECK_ARGUMENT(
        weight->ndim() == 2,
        "Embedding: weight must be a 2D tensor."
    );

    CHECK_ARGUMENT(
        out->ndim() == 2,
        "Embedding: out must be a 2D tensor."
    );

    CHECK_ARGUMENT(
        index->dtype() == LLAISYS_DTYPE_I64,
        "Embedding: index must have int64 dtype."
    );

    // embedding 输出只是 weight 行的副本，dtype 必须一致
    CHECK_ARGUMENT(
        out->dtype() == weight->dtype(),
        "Embedding: out and weight must have the same dtype."
    );

    CHECK_ARGUMENT(
        index->shape()[0] == out->shape()[0],
        "Embedding: out.shape[0] must equal index.shape[0]."
    );

    CHECK_ARGUMENT(
        out->shape()[1] == weight->shape()[1],
        "Embedding: out.shape[1] must equal weight.shape[1]."
    );

    ASSERT(
        out->isContiguous()
            && index->isContiguous()
            && weight->isContiguous(),
        "Embedding: all tensors must be contiguous."
    );

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            index->numel(),
            weight->shape()[0],
            weight->shape()[1],
            weight->elementSize()
        );
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            index->numel(),
            weight->shape()[0],
            weight->shape()[1],
            weight->elementSize()
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(
            out->data(),
            index->data(),
            weight->data(),
            out->dtype(),
            index->numel(),
            weight->shape()[0],
            weight->shape()[1],
            llaisys::core::context().runtime().stream()
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
