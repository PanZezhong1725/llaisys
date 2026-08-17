#include "swiglu_nvidia.cuh"

#include "../../../device/nvidia/cuda_helpers.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {

template <class Scalar>
__global__ void applySiluGate(Scalar *output, const Scalar *gate, const Scalar *up, size_t count) {
    const size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += step) {
        const float g = device::nvidia::scalarToFloat(gate[i]);
        const float result = (g / (1.0f + expf(-g))) * device::nvidia::scalarToFloat(up[i]);
        output[i] = device::nvidia::floatToScalar<Scalar>(result);
    }
}

template <class Scalar>
void launch(std::byte *output, const std::byte *gate, const std::byte *up, size_t count, cudaStream_t stream) {
    if (count == 0) return;
    constexpr unsigned int block = 256;
    applySiluGate<<<device::nvidia::gridFor(count, block), block, 0, stream>>>(
        reinterpret_cast<Scalar *>(output), reinterpret_cast<const Scalar *>(gate), reinterpret_cast<const Scalar *>(up), count);
    device::nvidia::requireCuda(cudaGetLastError(), "swiglu kernel");
}

} // namespace

void swiglu(std::byte *output, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, size_t count, llaisysStream_t stream) {
    const cudaStream_t native_stream = device::nvidia::cudaStream(stream);
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launch<float>(output, gate, up, count, native_stream);
    case LLAISYS_DTYPE_F16: return launch<__half>(output, gate, up, count, native_stream);
    case LLAISYS_DTYPE_BF16: return launch<__nv_bfloat16>(output, gate, up, count, native_stream);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
