#include "rms_norm.hpp"
#include "../../../utils.hpp"
#include <cmath>
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t rows, size_t d) {
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