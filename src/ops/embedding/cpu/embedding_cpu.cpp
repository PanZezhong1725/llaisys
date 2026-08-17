#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {

void embedding(
    std::byte *output,
    const std::byte *indices,
    const std::byte *table,
    llaisysDataType_t dtype,
    size_t index_count,
    size_t row_count,
    size_t row_width) {
    const auto *rows = reinterpret_cast<const int64_t *>(indices);
    const size_t row_bytes = row_width * utils::dsize(dtype);
    for (size_t i = 0; i < index_count; ++i) {
        CHECK_ARGUMENT(rows[i] >= 0 && static_cast<size_t>(rows[i]) < row_count, "embedding index is out of range");
        std::memcpy(
            output + i * row_bytes,
            table + static_cast<size_t>(rows[i]) * row_bytes,
            row_bytes);
    }
}

} // namespace llaisys::ops::cpu
