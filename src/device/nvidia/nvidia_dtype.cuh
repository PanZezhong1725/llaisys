#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace llaisys::device::nvidia {

/*
    这个 .cuh 文件会被多个 .cu 文件同时 include，__device__ inline 可以避免 header 中函数定义产生重复符号问题
*/

template <typename T>
__device__ inline float to_float(T x);

template <>
__device__ inline float to_float<float>(float x) {
    return x;
}

template <>
__device__ inline float to_float<__half>(__half x) {
    return __half2float(x);
}

template <>
__device__ inline float to_float<__nv_bfloat16>(
    __nv_bfloat16 x
) {
    return __bfloat162float(x);
}


template <typename T>
__device__ inline T from_float(float x);

template <>
__device__ inline float from_float<float>(float x) {
    return x;
}

template <>
__device__ inline __half from_float<__half>(float x) {
    return __float2half_rn(x);
}

template <>
__device__ inline __nv_bfloat16
from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}

} // namespace llaisys::device::nvidia