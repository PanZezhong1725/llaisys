#pragma once

#include "../core/llaisys_core.hpp"
#include "../device/corex/corex_check.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llaisys::ops::corex {

template <typename T>
__device__ inline float toFloat(T value);

template <>
__device__ inline float toFloat<float>(float value) {
    return value;
}

template <>
__device__ inline float toFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float toFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ inline T fromFloat(float value);

template <>
__device__ inline float fromFloat<float>(float value) {
    return value;
}

template <>
__device__ inline __half fromFloat<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ inline __nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

inline cudaStream_t currentStream() {
    return reinterpret_cast<cudaStream_t>(
        llaisys::core::context().runtime().stream());
}

inline void checkKernel() {
    COREX_CHECK(cudaGetLastError());
}

constexpr int BLOCK_SIZE = 256;

} // namespace llaisys::ops::corex
