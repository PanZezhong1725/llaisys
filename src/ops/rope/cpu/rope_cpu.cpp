#include "rope_cpu.hpp"

#include "../../../utils.hpp"
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <sys/types.h>

template <typename T>
void rope_impl(
	T *out,
	const T *in,
	const std::int64_t *pos_ids,
	float theta,
	size_t seq_len,
	size_t n_heads,
	size_t head_dim
) {
   const size_t half_dim = head_dim / 2;

    /*
     * 先检查全部 position id。
     * 这样可以避免处理到一半才发现非法索引，导致 out 中只写入了部分结果。
	 *
	 * 为什么不在 src/ops/rope/op.cpp 中检查？
	 * 	pos_ids[seq] >= 0 是对底层数据内容的检查，需要实际访问设备内存；
	 * 而 op.cpp 应保持设备无关，不能假设 Tensor 数据可以被 CPU 直接解引用
     */
    for (size_t seq = 0; seq < seq_len; ++seq) {
        CHECK_ARGUMENT(
            pos_ids[seq] >= 0,
            "RoPE: position ids must be non-negative."
        );
    }

	// 对固定 j，inv_freq 与 seq、head 都无关，
    // 把 j 放在最外层，只需要计算 half_dim 次 pow
	for (size_t j = 0; j < half_dim; ++j) {
        /*
         * PyTorch 参考实现是：
         *      freqs = positions / (theta ** (2 * i / head_dim))
         * 也就是先计算正指数的幂，再执行除法
         *
         * 如果这里的实现：
         *      const float inv_freq = std::pow(theta, -exponent);
         *      const float phi = position * inv_freq;
         *  
         * position * std::pow(theta, -exponent) 和 position / std::pow(theta, exponent) 
         * 不保证得到完全相同的 F32 位模式
         */

        // half_dim：决定有多少对，以及前后两半的地址关系
		// head_dim：RoPE 公式中的完整向量维度 d
		// 公式里的 d 是 head_dim
		// float exponent = 2 * j / head_dim; 会得到 0（size_t -> float）v
		const float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);

		// const float inv_freq = std::pow(theta, -exponent);
        const float freq_scale = std::pow(theta, exponent);

		for (size_t seq = 0; seq < seq_len; ++seq) {
			const float postion = static_cast<const float>(pos_ids[seq]);
			const float phi = postion / freq_scale;
			
			const float cos_phi = std::cos(phi); 
			const float sin_phi = std::sin(phi);

			
            // 相同 token 的所有 attention head 使用相同 position id 和频率
			for (size_t head = 0; head < n_heads; ++head) {
				const size_t base = (seq * n_heads + head) * head_dim;
                const size_t a_idx = base + j;
                const size_t b_idx = base + half_dim + j;

				const float a = llaisys::utils::cast<float>(in[a_idx]);
				const float b = llaisys::utils::cast<float>(in[b_idx]);

				const float rotated_a = a * cos_phi - b * sin_phi;
                const float rotated_b = b * cos_phi + a * sin_phi;

                out[a_idx] = llaisys::utils::cast<T>(rotated_a);
                out[b_idx] = llaisys::utils::cast<T>(rotated_b);
			}
		}
	}
}

namespace llaisys::ops::cpu {
void rope(
	std::byte *out,
	const std::byte *in,
	const std::byte *pos_ids,
	float theta,
	llaisysDataType_t type,
    size_t seq_len,
    size_t n_heads,
	size_t head_dim
) {
	const auto *positions = reinterpret_cast<const std::int64_t *>(pos_ids);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_impl (
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            positions,
            theta,
            seq_len,
            n_heads,
			head_dim
        );
    case LLAISYS_DTYPE_F16:
        return rope_impl (
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            positions,
            theta,
            seq_len,
            n_heads,
			head_dim
        );
    case LLAISYS_DTYPE_BF16:
        return rope_impl (
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            positions,
            theta,
            seq_len,
            n_heads,
			head_dim
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }   
}
} // namespace llaisys::ops::cpu
