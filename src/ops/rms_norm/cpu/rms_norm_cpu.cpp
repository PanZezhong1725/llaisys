#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(
    T *out,
    const T *in,
    const T *weight,
    float eps,
    size_t rows,
    size_t hidden_size) {
    for (size_t row = 0; row < rows; ++row) {
        float square_sum = 0.0f;

        for (size_t column = 0; column < hidden_size; ++column) {
            const size_t index = row * hidden_size + column;
            const float value = llaisys::utils::cast<float>(in[index]);
            const T squared = llaisys::utils::cast<T>(value * value);
            square_sum += llaisys::utils::cast<float>(squared);
        }

        const T mean_square = llaisys::utils::cast<T>(
            square_sum / static_cast<float>(hidden_size));
        const T mean_with_eps = llaisys::utils::cast<T>(
            llaisys::utils::cast<float>(mean_square) + eps);
        const T inverse_rms = llaisys::utils::cast<T>(
            1.0f / std::sqrt(llaisys::utils::cast<float>(mean_with_eps)));

        for (size_t column = 0; column < hidden_size; ++column) {
            const size_t index = row * hidden_size + column;
            const float value = llaisys::utils::cast<float>(in[index]);
            const float scale = llaisys::utils::cast<float>(weight[column]);
            const T normalized = llaisys::utils::cast<T>(
                value * llaisys::utils::cast<float>(inverse_rms));
            out[index] = llaisys::utils::cast<T>(
                llaisys::utils::cast<float>(normalized) * scale);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    float eps,
    size_t rows,
    size_t hidden_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            eps, rows, hidden_size);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            eps, rows, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            eps, rows, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
