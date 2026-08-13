#include "embedding_suda.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace llaisys::ops::suda {

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                 size_t seq_len, size_t embed_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numel = seq_len * embed_dim;
    if (idx >= numel) {
        return;
    }
    size_t row = idx / embed_dim;
    size_t col = idx % embed_dim;
    int64_t token = index[row];
    out[idx] = weight[token * embed_dim + col];
}

template <typename T>
static void launch_embedding(std::byte *out, const int64_t *index, const std::byte *weight,
                             size_t seq_len, size_t embed_dim) {
    size_t numel = seq_len * embed_dim;
    const int block = 256;
    int grid = static_cast<int>((numel + block - 1) / block);
    embedding_kernel<T><<<grid, block>>>(reinterpret_cast<T *>(out),
                                         index,
                                         reinterpret_cast<const T *>(weight),
                                         seq_len, embed_dim);
}

void embedding(std::byte *out, const int64_t *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t seq_len, size_t embed_dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_embedding<float>(out, index, weight, seq_len, embed_dim);
    case LLAISYS_DTYPE_F16:
        return launch_embedding<__half>(out, index, weight, seq_len, embed_dim);
    case LLAISYS_DTYPE_BF16:
        return launch_embedding<__nv_bfloat16>(out, index, weight, seq_len, embed_dim);
    default:
        break;
    }
}

} // namespace llaisys::ops::suda