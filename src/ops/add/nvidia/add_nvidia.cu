#include "add_nvidia.cuh"

#include "../../../device/nvidia/cuda_helpers.cuh"

namespace llaisys::ops::nvidia {
namespace {

template <class Scalar>
__global__ void addElements(Scalar *output, const Scalar *left, const Scalar *right, size_t count) {
    const size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += step) {
        const float sum = device::nvidia::scalarToFloat(left[i]) + device::nvidia::scalarToFloat(right[i]);
        output[i] = device::nvidia::floatToScalar<Scalar>(sum);
    }
}

template <class Scalar>
void launch(std::byte *output, const std::byte *left, const std::byte *right, size_t count, cudaStream_t stream) {
    if (count == 0) return;
    constexpr unsigned int block = 256;
    addElements<<<device::nvidia::gridFor(count, block), block, 0, stream>>>(
        reinterpret_cast<Scalar *>(output), reinterpret_cast<const Scalar *>(left), reinterpret_cast<const Scalar *>(right), count);
    device::nvidia::requireCuda(cudaGetLastError(), "add kernel");
}

} // namespace

void add(std::byte *output, const std::byte *left, const std::byte *right, llaisysDataType_t dtype, size_t count, llaisysStream_t stream) {
    const cudaStream_t native_stream = device::nvidia::cudaStream(stream);
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launch<float>(output, left, right, count, native_stream);
    case LLAISYS_DTYPE_F16: return launch<__half>(output, left, right, count, native_stream);
    case LLAISYS_DTYPE_BF16: return launch<__nv_bfloat16>(output, left, right, count, native_stream);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
