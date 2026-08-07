#include "op.hpp"
#include "../../utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/nvidia_ops.cuh"
#endif

namespace llaisys::ops {
template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                     size_t d, size_t dv, float scale) {
    size_t group = nh / nkvh;
    size_t offset = kvlen - qlen;
    std::vector<float> scores(kvlen);
    for (size_t qi = 0; qi < qlen; qi++) {
        size_t max_key = offset + qi;
        for (size_t h = 0; h < nh; h++) {
            size_t kvh = h / group;
            size_t qbase = (qi * nh + h) * d;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t kj = 0; kj < kvlen; kj++) {
                if (kj > max_key) {
                    scores[kj] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                size_t kbase = (kj * nkvh + kvh) * d;
                float score = 0.0f;
                for (size_t p = 0; p < d; p++) {
                    score += utils::cast<float>(q[qbase + p]) * utils::cast<float>(k[kbase + p]);
                }
                scores[kj] = score * scale;
                max_score = std::max(max_score, scores[kj]);
            }
            float denom = 0.0f;
            for (size_t kj = 0; kj <= max_key; kj++) {
                scores[kj] = std::exp(scores[kj] - max_score);
                denom += scores[kj];
            }
            size_t obase = (qi * nh + h) * dv;
            for (size_t p = 0; p < dv; p++) {
                float value = 0.0f;
                for (size_t kj = 0; kj <= max_key; kj++) {
                    size_t vbase = (kj * nkvh + kvh) * dv;
                    value += scores[kj] / denom * utils::cast<float>(v[vbase + p]);
                }
                out[obase + p] = utils::cast<T>(value);
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_ARGUMENT(attn_val->deviceType() == LLAISYS_DEVICE_CPU || attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA, "unsupported self_attention device");
    CHECK_ARGUMENT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && attn_val->ndim() == 3, "self_attention expects 3D tensors");
    CHECK_ARGUMENT(q->shape()[0] <= k->shape()[0] && q->shape()[1] % k->shape()[1] == 0, "invalid self_attention sequence or head sizes");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0] && k->shape()[1] == v->shape()[1] && k->shape()[2] == q->shape()[2], "invalid self_attention key/value shapes");
    std::vector<size_t> expected_shape{q->shape()[0], q->shape()[1], v->shape()[2]};
    CHECK_ARGUMENT(attn_val->shape() == expected_shape, "invalid self_attention output shape");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "self_attention requires contiguous tensors");
    size_t qlen = q->shape()[0], kvlen = k->shape()[0], nh = q->shape()[1], nkvh = k->shape()[1], d = q->shape()[2], dv = v->shape()[2];
    if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
#ifdef ENABLE_NVIDIA_API
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), qlen, kvlen, nh, nkvh, d, dv, scale);
#else
        EXCEPTION_UNSUPPORTED_DEVICE;
#endif
    }
    switch (attn_val->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val->data()), reinterpret_cast<const float *>(q->data()), reinterpret_cast<const float *>(k->data()), reinterpret_cast<const float *>(v->data()), qlen, kvlen, nh, nkvh, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<fp16_t *>(attn_val->data()), reinterpret_cast<const fp16_t *>(q->data()), reinterpret_cast<const fp16_t *>(k->data()), reinterpret_cast<const fp16_t *>(v->data()), qlen, kvlen, nh, nkvh, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<bf16_t *>(attn_val->data()), reinterpret_cast<const bf16_t *>(q->data()), reinterpret_cast<const bf16_t *>(k->data()), reinterpret_cast<const bf16_t *>(v->data()), qlen, kvlen, nh, nkvh, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
}
} // namespace llaisys::ops
