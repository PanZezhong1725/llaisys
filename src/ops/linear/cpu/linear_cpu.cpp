#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    size_t m_size,
    size_t n_size,
    size_t k_size) {
    for (size_t m = 0; m < m_size; ++m) {
        for (size_t n = 0; n < n_size; ++n) {
            float sum = bias == nullptr
                            ? 0.0f
                            : llaisys::utils::cast<float>(bias[n]);

            for (size_t k = 0; k < k_size; ++k) {
                const float in_value =
                    llaisys::utils::cast<float>(in[m * k_size + k]);
                const float weight_value =
                    llaisys::utils::cast<float>(weight[n * k_size + k]);
                sum += in_value * weight_value;
            }

            out[m * n_size + n] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t m,
    size_t n,
    size_t k) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            reinterpret_cast<const llaisys::fp16_t *>(bias),
            m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            reinterpret_cast<const llaisys::bf16_t *>(bias),
            m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
