#include "rms_norm_corex.cuh"

#include "../../corex_common.cuh"

#include <cmath>

namespace llaisys::ops::corex {
namespace {

template <typename T>
__global__ void normalizeRows(T *out, const T *in, const T *weight,
                              float eps, size_t width) {
    __shared__ float partial[BLOCK_SIZE];
    const size_t row = blockIdx.x;
    float local_sum = 0.0f;
    for (size_t column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = toFloat(in[row * width + column]);
        local_sum += value * value;
    }
    partial[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float factor = rsqrtf(partial[0] / static_cast<float>(width) + eps);
    for (size_t column = threadIdx.x; column < width; column += blockDim.x) {
        const size_t i = row * width + column;
        const T scaled = fromFloat<T>(toFloat(in[i]) * factor);
        out[i] = fromFloat<T>(toFloat(scaled) * toFloat(weight[column]));
    }
}

template <typename T>
void dispatchNorm(std::byte *out, const std::byte *in,
                  const std::byte *weight, float eps, size_t rows,
                  size_t width) {
    normalizeRows<<<static_cast<unsigned int>(rows), BLOCK_SIZE, 0,
                    currentStream()>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight), eps, width);
    checkKernel();
}

} // namespace

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, float eps, size_t rows,
              size_t hidden_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return dispatchNorm<float>(out, in, weight, eps, rows, hidden_size);
    case LLAISYS_DTYPE_F16:
        return dispatchNorm<__half>(out, in, weight, eps, rows, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return dispatchNorm<__nv_bfloat16>(out, in, weight, eps, rows,
                                           hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::corex
