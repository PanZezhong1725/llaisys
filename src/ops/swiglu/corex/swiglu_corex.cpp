#include "swiglu_corex.cuh"

#include "../../corex_common.cuh"

#include <cmath>

namespace llaisys::ops::corex {
namespace {

template <typename T>
__global__ void applySwiGlu(T *out, const T *gate, const T *up,
                            size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const float x = toFloat(gate[i]);
    const T activated = fromFloat<T>(x / (1.0f + expf(-x)));
    out[i] = fromFloat<T>(toFloat(up[i]) * toFloat(activated));
}

template <typename T>
void dispatchSwiGlu(std::byte *out, const std::byte *gate,
                    const std::byte *up, size_t count) {
    const int grid = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    applySwiGlu<<<grid, BLOCK_SIZE, 0, currentStream()>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up), count);
    checkKernel();
}

} // namespace

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return dispatchSwiGlu<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return dispatchSwiGlu<__half>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return dispatchSwiGlu<__nv_bfloat16>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::corex
