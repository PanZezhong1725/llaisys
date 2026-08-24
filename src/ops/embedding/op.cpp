#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.hpp"
#endif
#ifdef ENABLE_SUDA_API
#include "suda/embedding_suda.hpp"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    size_t seq_len = index->numel();
    size_t embed_dim = out->shape()[1];
    size_t vocab_size = weight->shape()[0];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              out->dtype(), seq_len, embed_dim, vocab_size, index->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              out->dtype(), seq_len, embed_dim, vocab_size, index->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(),
                                 out->dtype(), seq_len, embed_dim, vocab_size);
#endif
#ifdef ENABLE_SUDA_API
    case LLAISYS_DEVICE_SUDA:
        return suda::embedding(out->data(), index->data(), weight->data(),
                               out->dtype(), seq_len, embed_dim, vocab_size);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops


