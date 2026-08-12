#include "linear_cpu.hpp"
#include "../../../utils/matmul_cpu.hpp"

// 线性变换: out[M,N] = in[M,K] × weight[N,K]ᵀ + bias (可选)
template <typename T>
void linear_(const T *in, const T *weight, T *out, size_t M, size_t K, size_t N, const T *bias = nullptr) {
    llaisys::ops::cpu::matmul(out, in, weight, M, N, K, K, K, N);
    if (bias) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                out[i * N + j] = llaisys::utils::cast<T>(
                    llaisys::utils::cast<float>(out[i * N + j])
                    + llaisys::utils::cast<float>(bias[j]));
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias, llaisysDataType_t type, size_t /*numel*/) {
    const size_t in_rows = in->shape()[0];
    const size_t in_cols = in->shape()[1];
    const size_t out_cols = weight->shape()[0];

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<const float *>(in->data()),
                       reinterpret_cast<const float *>(weight->data()),
                       reinterpret_cast<float *>(out->data()),
                       in_rows, in_cols, out_cols,
                       bias ? reinterpret_cast<const float *>(bias->data()) : nullptr);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<const llaisys::bf16_t *>(in->data()),
                       reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                       reinterpret_cast<llaisys::bf16_t *>(out->data()),
                       in_rows, in_cols, out_cols,
                       bias ? reinterpret_cast<const llaisys::bf16_t *>(bias->data()) : nullptr);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<const llaisys::fp16_t *>(in->data()),
                       reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                       reinterpret_cast<llaisys::fp16_t *>(out->data()),
                       in_rows, in_cols, out_cols,
                       bias ? reinterpret_cast<const llaisys::fp16_t *>(bias->data()) : nullptr);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu