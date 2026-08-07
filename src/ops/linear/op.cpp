#include "op.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/nvidia_ops.cuh"
#endif

namespace llaisys::ops {
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = bias == nullptr ? 0.0f : utils::cast<float>(bias[j]);
            for (size_t p = 0; p < k; p++) {
                sum += utils::cast<float>(in[i * k + p]) * utils::cast<float>(weight[j * k + p]);
            }
            out[i * n + j] = utils::cast<T>(sum);
        }
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_ARGUMENT(out->deviceType() == LLAISYS_DEVICE_CPU || out->deviceType() == LLAISYS_DEVICE_NVIDIA, "unsupported linear device");
    CHECK_ARGUMENT(in->ndim() == 2 && weight->ndim() == 2 && out->ndim() == 2, "linear expects 2D tensors");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1] && out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0], "invalid linear shapes");
    CHECK_ARGUMENT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(), "linear dtype mismatch");
    if (bias) {
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == weight->shape()[0] && bias->dtype() == out->dtype(), "invalid linear bias");
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && (!bias || bias->isContiguous()), "linear requires contiguous tensors");
    size_t m = in->shape()[0], k = in->shape()[1], n = weight->shape()[0];
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
#ifdef ENABLE_NVIDIA_API
        return nvidia::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), m, k, n);
#else
        EXCEPTION_UNSUPPORTED_DEVICE;
#endif
    }
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(in->data()), reinterpret_cast<const float *>(weight->data()), bias ? reinterpret_cast<const float *>(bias->data()) : nullptr, m, k, n);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(in->data()), reinterpret_cast<const fp16_t *>(weight->data()), bias ? reinterpret_cast<const fp16_t *>(bias->data()) : nullptr, m, k, n);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(in->data()), reinterpret_cast<const bf16_t *>(weight->data()), bias ? reinterpret_cast<const bf16_t *>(bias->data()) : nullptr, m, k, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
