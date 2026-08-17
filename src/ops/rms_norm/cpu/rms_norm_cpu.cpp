#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(const T *in, const T *weight, T *out, size_t numel, float eps, size_t d) {
    if (numel == 0) {
        return;
    }

    size_t n = numel / d;

    for (size_t i = 0; i < n; ++i) {
        float sum_sq = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            float val = llaisys::utils::cast<float>(in[i * d + j]);
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / d + eps);
        for (size_t j = 0; j < d; ++j) {
            float normalized_val = llaisys::utils::cast<float>(in[i * d + j]) / rms;
            out[i * d + j] = llaisys::utils::cast<T>(normalized_val * llaisys::utils::cast<float>(weight[j]));
        }
    }
}

// out：输出 Y 你暂时可以假设输出是一个2D连续张量，不涉及广播。
// input：输入 X 你暂时可以假设输入是一个2D连续张量，不涉及广播。标准化沿输入张量的最后一个维度（即每一行，长度为 d ）执行。
// weight：权重 W 1D张量，与输入张量的一行长度相同。
// eps：小值 以避免除以零。
namespace llaisys::ops::cpu {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, llaisysDataType_t type, float eps) {
    size_t numel = in->numel();
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<const float *>(in->data()),
                         reinterpret_cast<const float *>(weight->data()),
                         reinterpret_cast<float *>(out->data()),
                         numel, eps, weight->numel());
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                         reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                         reinterpret_cast<llaisys::bf16_t *>(out->data()),
                         numel, eps, weight->numel());
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                         reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                         reinterpret_cast<llaisys::fp16_t *>(out->data()),
                         numel, eps, weight->numel());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu