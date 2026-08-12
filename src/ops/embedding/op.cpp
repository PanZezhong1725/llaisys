#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.hpp"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "icore/embedding_nvidia.hpp"
#endif
#ifdef ENABLE_MUSA_API
#include "musa/embedding_musa.hpp"
#endif

namespace llaisys::ops {
// 从weight（2-D）中复制index（1-D）中的行到output（2-D）。index必须是Int64类型（PyTorch中int的默认数据类型）。
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64 || index->dtype() == LLAISYS_DTYPE_I32,
           "Embedding: index tensor must be an integer tensor.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out, index, weight, weight->dtype(), index->shape()[0], weight->shape()[1]);
    }

    llaisys::core::context().setDevice(weight->deviceType(), out->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out, index, weight, weight->dtype(), index->shape()[0], weight->shape()[1]);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out, index, weight, weight->dtype(), index->shape()[0], weight->shape()[1]);
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::embedding(out, index, weight, weight->dtype(), index->shape()[0], weight->shape()[1]);
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return musa::embedding(out, index, weight, weight->dtype(), index->shape()[0], weight->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
