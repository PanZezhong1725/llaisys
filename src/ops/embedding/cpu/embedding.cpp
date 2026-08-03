#include "embedding.hpp"
#include "../../../utils.hpp"
#include <cstring>
namespace llaisys::ops::cpu {


void embedding(std::byte *out, const std::byte *index, const std::byte
    *weight, llaisysDataType_t type, size_t num_indices, size_t embd_dim) {
        //获取单位数据大小
        size_t esz = llaisys::utils::dsize(type);
        //要去除的数组
        const int64_t *index_ptr = reinterpret_cast<const int64_t *>(index);
        
        for (size_t i = 0; i < num_indices; i++) {
            std::memcpy(out + i * embd_dim * esz,
                 weight + index_ptr[i] * embd_dim * esz,
                 embd_dim * esz);
        }
}
}