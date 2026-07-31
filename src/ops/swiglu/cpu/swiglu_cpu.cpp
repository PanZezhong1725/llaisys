#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cstddef>
#include <cmath>

namespace llaisys::ops::cpu {
template <typename T>
void swiglu_impl(
	T *out,
	const T* gate,
	const T* up,
	size_t numel
) {
	for (size_t i = 0; i < numel; ++i) {
		const float gate_value = llaisys::utils::cast<float>(gate[i]);
		const float up_value = llaisys::utils::cast<float>(up[i]);
		float swiglu_value = 0.0f;
	
		/*
		 *	Silu(g) = g / (1 + e^(-g))
		 *  当 g 为负数，需要考虑 exp 运算导致 float32 溢出；exp(89)就回爆 float32
		 *
		 *  对于负数：
		 *	    g / (1 + e^(-g)) = g * e^(g) / (e^(g) + 1)
		 * 		e^(g) 任然是 exp(负值)
		*/
		if (gate_value > 0.0f) {
			swiglu_value = gate_value / (1.0f + std::exp(-gate_value));
		} else {
			const float exp_value = std::exp(gate_value);
			swiglu_value = (gate_value * exp_value) / (1.0f + exp_value);
		}

		out[i] = llaisys::utils::cast<T>(up_value * swiglu_value);
	}
}

void swiglu(
	std::byte *out,
	const std::byte *gate,
	const std::byte *up,
	llaisysDataType_t type,
	size_t numel
) {
	switch(type) {
	case LLAISYS_DTYPE_F32:
		return swiglu_impl(
			reinterpret_cast<float *>(out),
			reinterpret_cast<const float *>(gate),
			reinterpret_cast<const float *>(up),
			numel
		);
	case LLAISYS_DTYPE_BF16:
		return swiglu_impl(
			reinterpret_cast<llaisys::bf16_t *>(out),
			reinterpret_cast<const llaisys::bf16_t *>(gate),
			reinterpret_cast<const llaisys::bf16_t *>(up),
			numel
		);
	case LLAISYS_DTYPE_F16:
		return swiglu_impl(
			reinterpret_cast<llaisys::fp16_t *>(out),
			reinterpret_cast<const llaisys::fp16_t *>(gate),
			reinterpret_cast<const llaisys::fp16_t *>(up),
			numel
		);
	default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    } 
}
} // llaisys::ops::cpu