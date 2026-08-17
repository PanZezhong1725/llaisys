#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../cpu/cpu_utils.hpp"
#include "cpu/embedding_cpu.hpp"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "../nvidia/nvidia_ops.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1, "Embedding index must be a 1D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding weight must be a 2D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "Embedding output must be a 2D tensor");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding index must use int64");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(cpu::is_supported_float(weight->dtype()), "Embedding only supports floating point weights");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0] && out->shape()[1] == weight->shape()[1],
                   "Embedding output shape is incompatible with index and weight");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->shape()[0],
                              weight->shape()[0], weight->shape()[1], weight->elementSize());
    }
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->shape()[0],
                                 weight->shape()[0], weight->shape()[1]);
    }
#endif
#ifdef ENABLE_ILUVATAR_API
    if (out->deviceType() == LLAISYS_DEVICE_ILUVATAR) {
        return nvidia::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->shape()[0],
                                 weight->shape()[0], weight->shape()[1]);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
