#include "argmax_nvidia.cuh"

#include "../../nvidia_common.cuh"

#include <cfloat>
#include <cstdint>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void argmaxKernel(int64_t *max_idx, T *max_val, const T *vals,
                             size_t numel) {
    __shared__ float values[THREADS];
    __shared__ int64_t indices[THREADS];
    float best_value = -FLT_MAX;
    int64_t best_index = static_cast<int64_t>(numel);
    for (size_t index = threadIdx.x; index < numel; index += blockDim.x) {
        const float value = toFloat(vals[index]);
        if (value > best_value
            || (value == best_value
                && static_cast<int64_t>(index) < best_index)) {
            best_value = value;
            best_index = static_cast<int64_t>(index);
        }
    }
    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            const float other_value = values[threadIdx.x + offset];
            const int64_t other_index = indices[threadIdx.x + offset];
            if (other_value > values[threadIdx.x]
                || (other_value == values[threadIdx.x]
                    && other_index < indices[threadIdx.x])) {
                values[threadIdx.x] = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        max_idx[0] = indices[0];
        max_val[0] = vals[indices[0]];
    }
}

template <typename T>
void launch(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            size_t numel) {
    argmaxKernel<<<1, THREADS, 0, currentStream()>>>(
        reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val), reinterpret_cast<const T *>(vals),
        numel);
    checkKernelLaunch();
}

} // namespace

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(max_idx, max_val, vals, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
