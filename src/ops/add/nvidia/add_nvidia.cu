#include "add_nvidia.cuh"

#include "../../nvidia_common.cuh"

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void addKernel(T *out, const T *a, const T *b, size_t numel) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        out[index] = fromFloat<T>(toFloat(a[index]) + toFloat(b[index]));
    }
}

template <typename T>
void launch(std::byte *out, const std::byte *a, const std::byte *b,
            size_t numel) {
    const int blocks = static_cast<int>((numel + THREADS - 1) / THREADS);
    addKernel<<<blocks, THREADS, 0, currentStream()>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b), numel);
    checkKernelLaunch();
}

} // namespace

void add(std::byte *out, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, a, b, numel);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, a, b, numel);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, a, b, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
