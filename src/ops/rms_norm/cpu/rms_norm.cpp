#include "rms_norm.hpp"
#include "../../../utils.hpp"
#include <cmath>
template <typename T>
void rms_norm_(
    T *out,             // 输出张量，形状为 [rows, d]
    const T *in,        // 输入张量，形状为 [rows, d]
    const T *weight,    // 可学习缩放权重，形状为 [d]
    float eps,          // 防止除零的稳定项 epsilon
    size_t rows,        // 输入向量的行数，例如 token 数量
    size_t d             // 每一行的特征维度，例如 hidden_size
) {
    // 对第 m 行，RMSNorm 公式为：
    // mean_square_m = (1 / d) * Σ_{k=0}^{d-1} in[m, k]^2
        // rms_m = sqrt(mean_square_m + eps)
    // out[m, k] = weight[k] * in[m, k] / rms_m
    for (size_t m = 0; m < rows; m++) {
        float mean_square = 0.0f;
        for (size_t k = 0; k < d; k++) {
            mean_square += llaisys::utils::cast<float>(in[m * d + k]) * llaisys::utils::cast<float>(in[m * d + k]);
        }
        mean_square /= static_cast<float>(d);
        float denom = std::sqrt(mean_square + eps);
        for (size_t k = 0; k < d; k++) {
            out[m * d + k] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(weight[k]) * llaisys::utils::cast<float>(in[m * d + k]) / denom);
        }
        
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t type, size_t rows, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), eps, rows, d);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), eps, rows, d);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), eps, rows, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu