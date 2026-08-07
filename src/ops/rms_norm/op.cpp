#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/nvidia_ops.cuh"
#endif

namespace llaisys::ops {
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t d, float eps) {
    for (size_t row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (size_t i = 0; i < d; i++) {
            float x = utils::cast<float>(in[row * d + i]);
            sum += x * x;
        }
        float inv_rms = 1.0f / std::sqrt(sum / static_cast<float>(d) + eps);
        for (size_t i = 0; i < d; i++) {
            out[row * d + i] = utils::cast<T>(utils::cast<float>(in[row * d + i]) * inv_rms * utils::cast<float>(weight[i]));
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_ARGUMENT(out->deviceType() == LLAISYS_DEVICE_CPU || out->deviceType() == LLAISYS_DEVICE_NVIDIA, "unsupported rms_norm device");
    CHECK_ARGUMENT(in->ndim() == 2 && out->shape() == in->shape() && weight->ndim() == 1 && weight->shape()[0] == in->shape()[1], "invalid rms_norm shapes");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "rms_norm requires contiguous tensors");
    size_t rows = in->shape()[0], d = in->shape()[1];
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
#ifdef ENABLE_NVIDIA_API
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), rows, d, eps);
#else
        EXCEPTION_UNSUPPORTED_DEVICE;
#endif
    }
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(in->data()), reinterpret_cast<const float *>(weight->data()), rows, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(in->data()), reinterpret_cast<const fp16_t *>(weight->data()), rows, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(in->data()), reinterpret_cast<const bf16_t *>(weight->data()), rows, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
