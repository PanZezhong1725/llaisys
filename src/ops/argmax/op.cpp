#include "op.hpp"
#include "nvidia/argmax_cuda.cuh"
#include "iluvatar/argmax_iluvatar.cuh"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be int64.");

    // 当前只支持一维输入
    ASSERT(vals->ndim() == 1, "Argmax: vals must be 1D.");
    CHECK_SAME_SHAPE(max_idx->shape(), max_val->shape(), std::vector<size_t>{1});

    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "Argmax: all tensors must be contiguous.");
    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(max_idx->deviceType(), max_idx->deviceId());

    switch (max_idx->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        cuda::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        iluvatar::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
}