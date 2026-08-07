#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/nvidia_ops.cuh"
#endif

namespace llaisys::ops {
template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float g = utils::cast<float>(gate[i]);
        out[i] = utils::cast<T>(utils::cast<float>(up[i]) * g / (1.0f + std::exp(-g)));
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_ARGUMENT(out->deviceType() == LLAISYS_DEVICE_CPU || out->deviceType() == LLAISYS_DEVICE_NVIDIA, "unsupported swiglu device");
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "swiglu requires contiguous tensors");
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
#ifdef ENABLE_NVIDIA_API
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
#else
        EXCEPTION_UNSUPPORTED_DEVICE;
#endif
    }
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(gate->data()), reinterpret_cast<const float *>(up->data()), out->numel());
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(gate->data()), reinterpret_cast<const fp16_t *>(up->data()), out->numel());
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(gate->data()), reinterpret_cast<const bf16_t *>(up->data()), out->numel());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
