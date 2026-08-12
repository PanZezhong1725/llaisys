#include "embedding_cpu.hpp"

#include "../../cpu/cpu_utils.hpp"

#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {

    void embedding(std::byte *out,
                   const std::byte *index,
                   const std::byte *weight,
                   llaisysDataType_t dtype,
                   size_t index_count,
                   size_t vocabulary_size,
                     size_t embedding_dim,
                     size_t element_size) {
    const auto  *indices = reinterpret_cast<const int64_t *>(index);
    for (size_t row = 0;row < index_count; ++row) {
        const int64_t index_value = indices[row];
        CHECK_ARGUMENT(index_value >= 0 && static_cast<size_t>(index_value) < vocabulary_size,
                       "Embedding index out of bounds");
        const size_t row_offset = static_cast<size_t>(index_value) * embedding_dim * element_size;
        const std::byte *src = weight + row_offset;
        std::byte *dst = out + row * embedding_dim * element_size;
        std::memcpy(dst, src, embedding_dim * element_size);
    }
    (void)dtype;
    }
}
