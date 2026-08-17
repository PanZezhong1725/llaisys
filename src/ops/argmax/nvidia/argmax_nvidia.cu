#include "argmax_nvidia.cuh"

#include "../../../device/nvidia/cuda_helpers.cuh"

#include <cstdint>
#include <limits>

namespace llaisys::ops::nvidia {
namespace {

template <class Scalar>
__global__ void reduceMaximum(int64_t *output_index, Scalar *output_value, const Scalar *input, size_t count) {
    __shared__ float values[256];
    __shared__ size_t indices[256];

    float local_value = -3.402823466e+38F;
    size_t local_index = count;
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        const float candidate = device::nvidia::scalarToFloat(input[i]);
        if (candidate > local_value || (candidate == local_value && i < local_index)) {
            local_value = candidate;
            local_index = i;
        }
    }
    values[threadIdx.x] = local_value;
    indices[threadIdx.x] = local_index;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value = values[threadIdx.x + stride];
            const size_t other_index = indices[threadIdx.x + stride];
            if (other_value > values[threadIdx.x]
                || (other_value == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
                values[threadIdx.x] = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output_index = static_cast<int64_t>(indices[0]);
        *output_value = input[indices[0]];
    }
}

template <class Scalar>
void launch(std::byte *index, std::byte *value, const std::byte *input, size_t count, cudaStream_t stream) {
    reduceMaximum<<<1, 256, 0, stream>>>(reinterpret_cast<int64_t *>(index), reinterpret_cast<Scalar *>(value), reinterpret_cast<const Scalar *>(input), count);
    device::nvidia::requireCuda(cudaGetLastError(), "argmax kernel");
}

} // namespace

void argmax(std::byte *index, std::byte *value, const std::byte *input, llaisysDataType_t dtype, size_t count, llaisysStream_t stream) {
    CHECK_ARGUMENT(count != 0, "argmax requires a non-empty input");
    const cudaStream_t native_stream = device::nvidia::cudaStream(stream);
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launch<float>(index, value, input, count, native_stream);
    case LLAISYS_DTYPE_F16: return launch<__half>(index, value, input, count, native_stream);
    case LLAISYS_DTYPE_BF16: return launch<__nv_bfloat16>(index, value, input, count, native_stream);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
