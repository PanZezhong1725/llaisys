#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {


/*
 * embedding 只是复制整行，不需要进行浮点运算，可以直接按字节复制
 * 就不需要进行类型转换
*/
void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim,
    size_t element_size
) {
    const auto *indices = reinterpret_cast<const std::int64_t *>(index);

    // 一整行占用的字节数
    const size_t row_bytes = embedding_dim * element_size;

    for (size_t i = 0; i < num_indices; ++i) {
        const std::int64_t row = indices[i];

        // 先检查 row >= 0，再转换为 size_t
        // 负的 int64_t 转为 size_t 后会变成一个巨大的无符号整数
        CHECK_ARGUMENT(
            row >= 0
                && static_cast<size_t>(row) < num_embeddings,
            "Embedding: index is out of range."
        );

        const auto *src = weight + static_cast<size_t>(row) * row_bytes;

        auto *dst = out + i * row_bytes;

        std::memcpy(dst, src, row_bytes);
    }
}

} // namespace llaisys::ops::cpu