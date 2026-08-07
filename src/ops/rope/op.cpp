#include "op.hpp"
#include "../../utils.hpp"
#include <cmath>
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/nvidia_ops.cuh"
#endif

namespace llaisys::ops {
template <typename T>
void rope_(T *out, const T *in, const int64_t *pos, size_t seq, size_t heads, size_t d, float theta) {
    size_t half = d / 2;
    for (size_t s = 0; s < seq; s++) {
        for (size_t h = 0; h < heads; h++) {
            size_t base = (s * heads + h) * d;
            float position = static_cast<float>(pos[s]);
            for (size_t j = 0; j < half; j++) {
                float angle = position / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(d));
                float c = std::cos(angle), sn = std::sin(angle);
                float a = utils::cast<float>(in[base + j]);
                float b = utils::cast<float>(in[base + half + j]);
                out[base + j] = utils::cast<T>(a * c - b * sn);
                out[base + half + j] = utils::cast<T>(b * c + a * sn);
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_ARGUMENT(out->deviceType() == LLAISYS_DEVICE_CPU || out->deviceType() == LLAISYS_DEVICE_NVIDIA, "unsupported rope device");
    CHECK_ARGUMENT(in->ndim() == 3 && out->shape() == in->shape() && pos_ids->ndim() == 1 && pos_ids->shape()[0] == in->shape()[0], "invalid rope shapes");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0 && pos_ids->dtype() == LLAISYS_DTYPE_I64, "invalid rope dimensions or position dtype");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "rope requires contiguous tensors");
    const auto *pos = reinterpret_cast<const int64_t *>(pos_ids->data());
    size_t seq = in->shape()[0], heads = in->shape()[1], d = in->shape()[2];
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
#ifdef ENABLE_NVIDIA_API
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seq, heads, d, theta);
#else
        EXCEPTION_UNSUPPORTED_DEVICE;
#endif
    }
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(in->data()), pos, seq, heads, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(in->data()), pos, seq, heads, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(in->data()), pos, seq, heads, d, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
