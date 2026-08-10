#include "embedding_nvidia.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *indices, const T *weight,
                                 size_t seq_len, size_t embed_dim, size_t vocab_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * embed_dim) {
        return;
    }
    size_t row = idx / embed_dim;
    size_t col = idx % embed_dim;
    int64_t token = indices[row];
    if (token >= 0 && token < static_cast<int64_t>(vocab_size)) {
        out[idx] = weight[token * embed_dim + col];
    } else {
        out[idx] = static_cast<T>(0);
    }
}

template <typename T>
static void launch_embedding(std::byte *out, const std::byte *indices, const std::byte *weight,
                             size_t seq_len, size_t embed_dim, size_t vocab_size) {
    size_t numel = seq_len * embed_dim;
    const int block = 256;
    int grid = static_cast<int>((numel + block - 1) / block);
    embedding_kernel<T><<<grid, block>>>(reinterpret_cast<T *>(out),
                                         reinterpret_cast<const int64_t *>(indices),
                                         reinterpret_cast<const T *>(weight),
                                         seq_len, embed_dim, vocab_size);
}

void embedding(std::byte *out, const std::byte *indices, const std::byte *weight,
               llaisysDataType_t dtype, size_t seq_len, size_t embed_dim, size_t vocab_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_embedding<float>(out, indices, weight, seq_len, embed_dim, vocab_size);
    case LLAISYS_DTYPE_F16:
        return launch_embedding<__half>(out, indices, weight, seq_len, embed_dim, vocab_size);
    case LLAISYS_DTYPE_BF16:
        return launch_embedding<__nv_bfloat16>(out, indices, weight, seq_len, embed_dim, vocab_size);
    default:
        break;
    }
}

} // namespace llaisys::ops::nvidia
