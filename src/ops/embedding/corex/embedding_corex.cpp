#include "embedding_corex.cuh"

#include "../../corex_common.cuh"

#include <cstdint>

namespace llaisys::ops::corex {
namespace {

__global__ void gatherRows(std::byte *out, const int64_t *indices,
                           const std::byte *table, size_t token_count,
                           size_t row_count, size_t width,
                           size_t element_size) {
    const size_t output_element = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t output_count = token_count * width;
    if (output_element >= output_count) {
        return;
    }
    const size_t token = output_element / width;
    const size_t column = output_element % width;
    const int64_t source_row = indices[token];
    if (source_row < 0 || static_cast<size_t>(source_row) >= row_count) {
        return;
    }
    const size_t dst = output_element * element_size;
    const size_t src =
        (static_cast<size_t>(source_row) * width + column) * element_size;
    for (size_t i = 0; i < element_size; ++i) {
        out[dst + i] = table[src + i];
    }
}

} // namespace

void embedding(std::byte *out, const std::byte *index,
               const std::byte *weight, size_t token_count,
               size_t vocab_size, size_t hidden_size, size_t element_size) {
    const size_t count = token_count * hidden_size;
    const int grid = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    gatherRows<<<grid, BLOCK_SIZE, 0, currentStream()>>>(
        out, reinterpret_cast<const int64_t *>(index), weight, token_count,
        vocab_size, hidden_size, element_size);
    checkKernel();
}
} // namespace llaisys::ops::corex
