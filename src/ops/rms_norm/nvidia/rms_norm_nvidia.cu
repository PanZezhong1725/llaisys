#include "rms_norm_nvidia.cuh"

#include "../../nvidia_common.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void rmsNormKernel(T *out, const T *in, const T *weight,
                              float eps, size_t hidden_size) {
    __shared__ float reduction[THREADS];
    const size_t row = blockIdx.x;
    float square_sum = 0.0f;
    for (size_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
        const float value = toFloat(in[row * hidden_size + column]);
        square_sum += value * value;
    }
    reduction[threadIdx.x] = square_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(
        reduction[0] / static_cast<float>(hidden_size) + eps);
    for (size_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
        const size_t index = row * hidden_size + column;
        const T normalized = fromFloat<T>(toFloat(in[index]) * inverse_rms);
        out[index] = fromFloat<T>(toFloat(normalized) * toFloat(weight[column]));
    }
}

template <typename T>
void launch(std::byte *out, const std::byte *in, const std::byte *weight,
            float eps, size_t rows, size_t hidden_size) {
    rmsNormKernel<<<static_cast<unsigned int>(rows), THREADS, 0,
                    currentStream()>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight), eps, hidden_size);
    checkKernelLaunch();
}

} // namespace

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, float eps, size_t rows,
              size_t hidden_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, in, weight, eps, rows, hidden_size);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, in, weight, eps, rows, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, in, weight, eps, rows, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
