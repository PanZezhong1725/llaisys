#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void embedding_(T *out, const std::int64_t *index, const T *weight, size_t num_rows, size_t num_cols) {
    for (size_t i = 0; i < num_rows; i++) {
        std::int64_t idx = index[i];
        for (size_t j = 0; j < num_cols; j++) {
            out[i * num_cols + j] = weight[idx * num_cols + j];
        }
    }
}

// 从weight（2-D）中复制index（1-D）中的行到output（2-D）。index必须是Int64类型（PyTorch中int的默认数据类型）。
namespace llaisys::ops::cpu {
void embedding(tensor_t out, tensor_t index, tensor_t weight, llaisysDataType_t type, size_t num_rows, size_t num_cols) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out->data()),
                          reinterpret_cast<const std::int64_t *>(index->data()),
                          reinterpret_cast<const float *>(weight->data()),
                          num_rows, num_cols);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out->data()),
                          reinterpret_cast<const std::int64_t *>(index->data()),
                          reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
                          num_rows, num_cols);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out->data()),
                          reinterpret_cast<const std::int64_t *>(index->data()),
                          reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
                          num_rows, num_cols);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu