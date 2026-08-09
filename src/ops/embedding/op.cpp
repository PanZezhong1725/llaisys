#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);

    CHECK_ARGUMENT(index->ndim() == 1,
                   "Embedding: index must be one-dimensional.");
    CHECK_ARGUMENT(weight->ndim() == 2,
                   "Embedding: weight must be two-dimensional.");
    CHECK_ARGUMENT(out->ndim() == 2,
                   "Embedding: out must be two-dimensional.");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0],
                   "Embedding: out and index must have the same first dimension.");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[1],
                   "Embedding: out and weight must have the same second dimension.");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64,
                   "Embedding: index must be int64.");
    CHECK_ARGUMENT(weight->dtype() == out->dtype(),
                   "Embedding: weight and out dtype mismatch.");

    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            index->shape()[0],
            weight->shape()[0],
            weight->shape()[1],
            weight->elementSize());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            index->shape()[0],
            weight->shape()[0],
            weight->shape()[1],
            weight->elementSize());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(
            out->data(), index->data(), weight->data(), index->shape()[0],
            weight->shape()[0], weight->shape()[1], weight->elementSize());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
