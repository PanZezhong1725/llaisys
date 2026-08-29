#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        float sum_sq = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            float x = llaisys::utils::cast<float>(in[r * cols + c]);
            sum_sq += x * x;
        }
        float rms = std::sqrt(sum_sq / cols + eps);
        for (size_t c = 0; c < cols; c++) {
            float x = llaisys::utils::cast<float>(in[r * cols + c]);
            float w = llaisys::utils::cast<float>(weight[c]);
            out[r * cols + c] = llaisys::utils::cast<T>(w * x / rms);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t rows, size_t cols) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), eps, rows, cols);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), eps, rows, cols);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), eps, rows, cols);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
