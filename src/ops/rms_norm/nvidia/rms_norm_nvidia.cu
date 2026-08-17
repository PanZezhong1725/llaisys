#include "rms_norm_nvidia.cuh"

#include "../../../device/nvidia/cuda_helpers.cuh"

namespace llaisys::ops::nvidia {
namespace {

template <class Scalar>
__global__ void normalize(Scalar *output, const Scalar *input, const Scalar *weight, size_t rows, size_t width, float epsilon) {
    __shared__ float partial[256];
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        float local = 0.0f;
        for (size_t column = threadIdx.x; column < width; column += blockDim.x) {
            const float value = device::nvidia::scalarToFloat(input[row * width + column]);
            local += value * value;
        }
        partial[threadIdx.x] = local;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        const float inverse_rms = rsqrtf(partial[0] / static_cast<float>(width) + epsilon);
        for (size_t column = threadIdx.x; column < width; column += blockDim.x) {
            const float result = device::nvidia::scalarToFloat(input[row * width + column])
                               * inverse_rms
                               * device::nvidia::scalarToFloat(weight[column]);
            output[row * width + column] = device::nvidia::floatToScalar<Scalar>(result);
        }
        __syncthreads();
    }
}

template <class Scalar>
void launch(std::byte *output, const std::byte *input, const std::byte *weight, size_t rows, size_t width, float epsilon, cudaStream_t stream) {
    if (rows == 0) return;
    constexpr unsigned int block = 256;
    const unsigned int grid = static_cast<unsigned int>(rows < 65535 ? rows : 65535);
    normalize<<<grid, block, 0, stream>>>(reinterpret_cast<Scalar *>(output), reinterpret_cast<const Scalar *>(input), reinterpret_cast<const Scalar *>(weight), rows, width, epsilon);
    device::nvidia::requireCuda(cudaGetLastError(), "rms_norm kernel");
}

} // namespace

void rms_norm(std::byte *output, const std::byte *input, const std::byte *weight, llaisysDataType_t dtype, size_t rows, size_t width, float epsilon, llaisysStream_t stream) {
    const cudaStream_t native_stream = device::nvidia::cudaStream(stream);
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launch<float>(output, input, weight, rows, width, epsilon, native_stream);
    case LLAISYS_DTYPE_F16: return launch<__half>(output, input, weight, rows, width, epsilon, native_stream);
    case LLAISYS_DTYPE_BF16: return launch<__nv_bfloat16>(output, input, weight, rows, width, epsilon, native_stream);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
