#include "embedding_nvidia.cuh"

#include "../../nvidia_common.cuh"

#include <cstdint>

namespace llaisys::ops::nvidia {
namespace {

__global__ void embeddingKernel(
    std::byte *out, const int64_t *indices, const std::byte *weight,
    size_t token_count, size_t vocab_size, size_t hidden_size,
    size_t element_size) {
    const size_t element = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = token_count * hidden_size;
    if (element >= numel) {
        return;
    }
    const size_t token = element / hidden_size;
    const size_t column = element % hidden_size;
    const int64_t row = indices[token];
    if (row < 0 || static_cast<size_t>(row) >= vocab_size) {
        return;
    }
    const size_t dst_offset = element * element_size;
    const size_t src_offset =
        (static_cast<size_t>(row) * hidden_size + column) * element_size;
    for (size_t byte = 0; byte < element_size; ++byte) {
        out[dst_offset + byte] = weight[src_offset + byte];
    }
}

} // namespace

void embedding(std::byte *out, const std::byte *index,
               const std::byte *weight, size_t token_count,
               size_t vocab_size, size_t hidden_size, size_t element_size) {
    const size_t numel = token_count * hidden_size;
    const int blocks = static_cast<int>((numel + THREADS - 1) / THREADS);
    embeddingKernel<<<blocks, THREADS, 0, currentStream()>>>(
        out, reinterpret_cast<const int64_t *>(index), weight, token_count,
        vocab_size, hidden_size, element_size);
    checkKernelLaunch();
}
} // namespace llaisys::ops::nvidia
