#include "add_corex.cuh"

#include "../../corex_common.cuh"

namespace llaisys::ops::corex {
namespace {

template <typename T>
__global__ void elementwiseAdd(T *out, const T *a, const T *b, size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        out[i] = fromFloat<T>(toFloat(a[i]) + toFloat(b[i]));
    }
}

template <typename T>
void dispatchAdd(std::byte *out, const std::byte *a, const std::byte *b,
                 size_t count) {
    const int grid = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    elementwiseAdd<<<grid, BLOCK_SIZE, 0, currentStream()>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b), count);
    checkKernel();
}

} // namespace

void add(std::byte *out, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return dispatchAdd<float>(out, a, b, numel);
    case LLAISYS_DTYPE_F16:
        return dispatchAdd<__half>(out, a, b, numel);
    case LLAISYS_DTYPE_BF16:
        return dispatchAdd<__nv_bfloat16>(out, a, b, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::corex
