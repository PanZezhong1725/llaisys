#pragma once

#include <cstddef>

namespace llaisys::ops::cpu {

/*
 * 参数含义：
 *    out             输出数据地址
 *    index           Int64 索引数据地址
 *    weight          embedding table 数据地址
 *    num_indices     index 中的元素数量
 *    num_embeddings  weight 的行数
 *    embedding_dim   weight 的列数
 *    element_size    weight 每个元素占用的字节数
 */
void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim,
    size_t element_size
);

} //

