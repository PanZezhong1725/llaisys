#include "linear_cpu.hpp"
#include "../../../utils.hpp"
// in :[M,K], weight :[N,K], bias :[N], out :[M,N]
// M: in's first dimension, N: weight's first dimension, K: in's second dimension
// out = in * weight^T + bias

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K) {
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float acc = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    acc += llaisys::utils::cast<float>(in[m * K + k]) * llaisys::utils::cast<float>(weight[n * K + k]);
                }
                out[m * N + n] = llaisys::utils::cast<T>(acc);
            } else {
                T acc = 0;
                for (size_t k = 0; k < K; k++) {
                    acc += in[m * K + k] * weight[n * K + k];
                }
                out[m * N + n] = acc;
            }
        }
    }
    if (bias != nullptr) {
        for (size_t m = 0; m < M; m++) {
            for (size_t n = 0; n < N; n++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[m * N + n] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(out[m * N + n]) + llaisys::utils::cast<float>(bias[n]));
                } else {
                    out[m * N + n] += bias[n];
                }

            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type, size_t M, size_t N, size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu