#include "argmax_corex.cuh"

#include "../../corex_common.cuh"

#include <cfloat>
#include <cstdint>

namespace llaisys::ops::corex {
namespace {

template <typename T>
__global__ void reduceMaximum(int64_t *result_index, T *result_value,
                              const T *values, size_t count) {
    __shared__ float candidates[BLOCK_SIZE];
    __shared__ int64_t candidate_indices[BLOCK_SIZE];
    float local_value = -FLT_MAX;
    int64_t local_index = static_cast<int64_t>(count);
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        const float value = toFloat(values[i]);
        if (value > local_value
            || (value == local_value && static_cast<int64_t>(i) < local_index)) {
            local_value = value;
            local_index = static_cast<int64_t>(i);
        }
    }
    candidates[threadIdx.x] = local_value;
    candidate_indices[threadIdx.x] = local_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float rhs = candidates[threadIdx.x + stride];
            const int64_t rhs_index = candidate_indices[threadIdx.x + stride];
            if (rhs > candidates[threadIdx.x]
                || (rhs == candidates[threadIdx.x]
                    && rhs_index < candidate_indices[threadIdx.x])) {
                candidates[threadIdx.x] = rhs;
                candidate_indices[threadIdx.x] = rhs_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        result_index[0] = candidate_indices[0];
        result_value[0] = values[candidate_indices[0]];
    }
}

template <typename T>
void dispatchArgmax(std::byte *max_idx, std::byte *max_val,
                    const std::byte *vals, size_t count) {
    reduceMaximum<<<1, BLOCK_SIZE, 0, currentStream()>>>(
        reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals), count);
    checkKernel();
}

} // namespace

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return dispatchArgmax<float>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_F16:
        return dispatchArgmax<__half>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_BF16:
        return dispatchArgmax<__nv_bfloat16>(max_idx, max_val, vals, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::corex
