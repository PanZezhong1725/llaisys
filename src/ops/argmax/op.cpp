#include "op.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/nvidia_ops.cuh"
#endif

namespace llaisys::ops {
template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t n) {
    float best = utils::cast<float>(vals[0]);
    size_t best_idx = 0;
    for (size_t i = 1; i < n; i++) {
        float value = utils::cast<float>(vals[i]);
        if (value > best) {
            best = value;
            best_idx = i;
        }
    }
    *max_idx = static_cast<int64_t>(best_idx);
    *max_val = utils::cast<T>(best);
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_ARGUMENT(vals->deviceType() == LLAISYS_DEVICE_CPU || vals->deviceType() == LLAISYS_DEVICE_NVIDIA, "unsupported argmax device");
    CHECK_ARGUMENT(vals->ndim() == 1 && vals->numel() > 0, "argmax expects a non-empty 1D tensor");
    CHECK_ARGUMENT(max_idx->shape() == std::vector<size_t>{1}, "invalid argmax index output shape");
    CHECK_ARGUMENT(max_val->shape() == std::vector<size_t>{1}, "invalid argmax value output shape");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax index must be int64");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(), "argmax requires contiguous tensors");
    if (vals->deviceType() == LLAISYS_DEVICE_NVIDIA) {
#ifdef ENABLE_NVIDIA_API
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#else
        EXCEPTION_UNSUPPORTED_DEVICE;
#endif
    }
    auto *idx = reinterpret_cast<int64_t *>(max_idx->data());
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        return argmax_(idx, reinterpret_cast<float *>(max_val->data()), reinterpret_cast<const float *>(vals->data()), vals->numel());
    case LLAISYS_DTYPE_F16:
        return argmax_(idx, reinterpret_cast<fp16_t *>(max_val->data()), reinterpret_cast<const fp16_t *>(vals->data()), vals->numel());
    case LLAISYS_DTYPE_BF16:
        return argmax_(idx, reinterpret_cast<bf16_t *>(max_val->data()), reinterpret_cast<const bf16_t *>(vals->data()), vals->numel());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}
} // namespace llaisys::ops
