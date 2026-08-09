#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {
void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    size_t token_count,
    size_t vocab_size,
    size_t hidden_size,
    size_t element_size) {
    const auto *indices = reinterpret_cast<const int64_t *>(index);
    const size_t row_size = hidden_size * element_size;

    for (size_t i = 0; i < token_count; ++i) {
        const int64_t row = indices[i];
        CHECK_ARGUMENT(row >= 0 && static_cast<size_t>(row) < vocab_size,
                       "Embedding: index is out of bounds.");

        std::memcpy(
            out + i * row_size,
            weight + static_cast<size_t>(row) * row_size,
            row_size);
    }
}
} // namespace llaisys::ops::cpu
