#include "rope_nvidia.cuh"

#include "../../../device/nvidia/cuda_helpers.cuh"

#include <cstdint>

namespace llaisys::ops::nvidia {
namespace {

template <class Scalar>
__global__ void rotatePairs(Scalar *output, const Scalar *input, const int64_t *positions, size_t pairs, size_t heads, size_t width, float theta) {
    const size_t half = width / 2;
    const size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t pair_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; pair_index < pairs; pair_index += step) {
        const size_t pair = pair_index % half;
        const size_t token_head = pair_index / half;
        const size_t token = token_head / heads;
        const size_t base = token_head * width;
        const float exponent = 2.0f * static_cast<float>(pair) / static_cast<float>(width);
        const float angle = static_cast<float>(positions[token]) / powf(theta, exponent);
        float sine = 0.0f;
        float cosine = 0.0f;
        sincosf(angle, &sine, &cosine);
        const float first = device::nvidia::scalarToFloat(input[base + pair]);
        const float second = device::nvidia::scalarToFloat(input[base + half + pair]);
        output[base + pair] = device::nvidia::floatToScalar<Scalar>(first * cosine - second * sine);
        output[base + half + pair] = device::nvidia::floatToScalar<Scalar>(second * cosine + first * sine);
    }
}

template <class Scalar>
void launch(std::byte *output, const std::byte *input, const std::byte *positions, size_t sequence, size_t heads, size_t width, float theta, cudaStream_t stream) {
    const size_t pairs = sequence * heads * (width / 2);
    if (pairs == 0) return;
    constexpr unsigned int block = 256;
    rotatePairs<<<device::nvidia::gridFor(pairs, block), block, 0, stream>>>(
        reinterpret_cast<Scalar *>(output), reinterpret_cast<const Scalar *>(input), reinterpret_cast<const int64_t *>(positions), pairs, heads, width, theta);
    device::nvidia::requireCuda(cudaGetLastError(), "rope kernel");
}

} // namespace

void rope(std::byte *output, const std::byte *input, const std::byte *positions, llaisysDataType_t dtype, size_t sequence, size_t heads, size_t width, float theta, llaisysStream_t stream) {
    const cudaStream_t native_stream = device::nvidia::cudaStream(stream);
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launch<float>(output, input, positions, sequence, heads, width, theta, native_stream);
    case LLAISYS_DTYPE_F16: return launch<__half>(output, input, positions, sequence, heads, width, theta, native_stream);
    case LLAISYS_DTYPE_BF16: return launch<__nv_bfloat16>(output, input, positions, sequence, heads, width, theta, native_stream);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
