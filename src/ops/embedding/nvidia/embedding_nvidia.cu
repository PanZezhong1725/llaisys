#include "embedding_nvidia.cuh"

#include "../../../device/nvidia/cuda_helpers.cuh"

#include <cstdint>

namespace llaisys::ops::nvidia {
namespace {

template <class Scalar>
__global__ void gatherRows(Scalar *output, const int64_t *indices, const Scalar *table, size_t elements, size_t rows, size_t width) {
    const size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < elements; i += step) {
        const size_t item = i / width;
        const size_t column = i - item * width;
        const int64_t row = indices[item];
        output[i] = row >= 0 && static_cast<size_t>(row) < rows
            ? table[static_cast<size_t>(row) * width + column]
            : Scalar{};
    }
}

template <class Scalar>
void launch(std::byte *output, const std::byte *indices, const std::byte *table, size_t count, size_t rows, size_t width, cudaStream_t stream) {
    const size_t elements = count * width;
    if (elements == 0) return;
    constexpr unsigned int block = 256;
    gatherRows<<<device::nvidia::gridFor(elements, block), block, 0, stream>>>(
        reinterpret_cast<Scalar *>(output), reinterpret_cast<const int64_t *>(indices), reinterpret_cast<const Scalar *>(table), elements, rows, width);
    device::nvidia::requireCuda(cudaGetLastError(), "embedding kernel");
}

} // namespace

void embedding(std::byte *output, const std::byte *indices, const std::byte *table, llaisysDataType_t dtype, size_t index_count, size_t row_count, size_t row_width, llaisysStream_t stream) {
    const cudaStream_t native_stream = device::nvidia::cudaStream(stream);
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launch<float>(output, indices, table, index_count, row_count, row_width, native_stream);
    case LLAISYS_DTYPE_F16: return launch<__half>(output, indices, table, index_count, row_count, row_width, native_stream);
    case LLAISYS_DTYPE_BF16: return launch<__nv_bfloat16>(output, indices, table, index_count, row_count, row_width, native_stream);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
