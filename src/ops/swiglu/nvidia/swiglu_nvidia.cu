#include "swiglu_nvidia.cuh"

#include "../../nvidia_common.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up,
                             size_t numel) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }
    const float gate_value = toFloat(gate[index]);
    const T silu = fromFloat<T>(gate_value / (1.0f + expf(-gate_value)));
    out[index] = fromFloat<T>(toFloat(up[index]) * toFloat(silu));
}

template <typename T>
void launch(std::byte *out, const std::byte *gate, const std::byte *up,
            size_t numel) {
    const int blocks = static_cast<int>((numel + THREADS - 1) / THREADS);
    swigluKernel<<<blocks, THREADS, 0, currentStream()>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up), numel);
    checkKernelLaunch();
}

} // namespace

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
