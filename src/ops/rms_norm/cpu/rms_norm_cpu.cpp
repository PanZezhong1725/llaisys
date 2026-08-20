#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"
#include <cstddef>
#include <cmath>

template <typename T>
void rms_norm_impl(
	T *out,
	const T *in,
	const T *weight,
	float eps,
    size_t num_groups,
    size_t norm_size
) {
    for (size_t dim = 0; dim < num_groups; ++dim) {
        float sum_of_squares = 0.0f;

        // 计算当前 dim 的正确内存起始偏移
        size_t start = dim * norm_size;

        // 计算当前维度的平方和
        for (size_t i = 0; i < norm_size; ++i) {
            float val = 0.0f;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[start + i]);
            } else {
                val = in[start + i] ;
            }
            sum_of_squares += val * val;
        }

        // 计算均方根倒数，避免在下面的循环中多次除法运算
        float mean_of_squares = sum_of_squares / static_cast<float>(norm_size);
        float inv_rms = 1.0f / sqrtf(mean_of_squares + eps);

        // 归一化，并乘以权重
        for (size_t i =0; i < norm_size; ++i) {
            float val = 0.0f;
            float scale = 0.0f;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in[start + i]);
                scale = llaisys::utils::cast<float>(weight[i]);
            } else {
                val = in[start + i];
                scale = weight[i];
            }
            out[start + i] = llaisys::utils::cast<T>((val * scale) * inv_rms);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(
	std::byte *out,
	const std::byte *in,
	const std::byte *weight,
	float eps,
	llaisysDataType_t type,
    size_t num_groups,
    size_t norm_size
) {
     switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl (
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            eps,
            num_groups,
            norm_size
        );
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl (
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            eps,
            num_groups,
            norm_size
        );
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl (
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            eps,
            num_groups,
            norm_size
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }   
}
} // namespace llaisys::ops::cpu
