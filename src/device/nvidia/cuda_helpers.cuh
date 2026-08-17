#pragma once

#include "../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

inline void requireCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

inline cudaStream_t cudaStream(llaisysStream_t stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

inline unsigned int gridFor(size_t work, unsigned int block_size = 256) {
    if (work == 0) {
        return 0;
    }
    const size_t wanted = (work + block_size - 1) / block_size;
    return static_cast<unsigned int>(wanted < 65535 ? wanted : 65535);
}

template <class Scalar>
__device__ inline float scalarToFloat(Scalar value);

template <>
__device__ inline float scalarToFloat<float>(float value) { return value; }

template <>
__device__ inline float scalarToFloat<__half>(__half value) { return __half2float(value); }

template <>
__device__ inline float scalarToFloat<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }

template <class Scalar>
__device__ inline Scalar floatToScalar(float value);

template <>
__device__ inline float floatToScalar<float>(float value) { return value; }

template <>
__device__ inline __half floatToScalar<__half>(float value) { return __float2half(value); }

template <>
__device__ inline __nv_bfloat16 floatToScalar<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

} // namespace llaisys::device::nvidia
