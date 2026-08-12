#include "embedding_iluvatar.cuh"

#include "../../../utils.hpp"
#include "../../../device/iluvatar/iluvatar_utils.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace llaisys::ops::iluvatar {

namespace {

template <typename T>
__global__ void embedding_kernel(
    T *out,
    const std::int64_t *indices,
    const T *weight,
    size_t seq_len,
    size_t vocab_size,
    size_t hidden_size
) {
    const size_t token = static_cast<size_t>(blockIdx.x);

    if (token >= seq_len) {
        return;
    }

    __shared__ std::int64_t shared_token_id;

    if (threadIdx.x == 0) {
        shared_token_id = indices[token];
    }

    __syncthreads();

    const std::int64_t token_id = shared_token_id;

    if ( token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
        return;
    }

    const T *src = weight + static_cast<size_t>(token_id) * hidden_size;

    T *dst = out + token * hidden_size;

    for (size_t dim = threadIdx.x; dim < hidden_size; dim += blockDim.x) {
        dst[dim] = src[dim];
    }
}

} // namespace

void embedding(
    std::byte *out,
    const std::byte *indices,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t seq_len,
    size_t vocab_size,
    size_t hidden_size,
    llaisysStream_t stream_
) {
    if (seq_len == 0 || hidden_size == 0) {
        return;
    }

    constexpr int BLOCK_SIZE = 256;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);

    const auto *indices_i64 = reinterpret_cast<const std::int64_t *>(indices);

    const dim3 grid(static_cast<unsigned int>(seq_len));

    const dim3 block(BLOCK_SIZE);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<float *>(out),
                indices_i64,
                reinterpret_cast<const float *>(weight),
                seq_len,
                vocab_size,
                hidden_size
            );
        break;
    // Embedding 不关心浮点语义，只关心元素大小，纯粹的数据搬运角度
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<std::uint16_t>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<std::uint16_t *>(out),
                indices_i64,
                reinterpret_cast<const std::uint16_t *>(weight),
                seq_len,
                vocab_size,
                hidden_size
            );
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    ILUVATAR_CUDA_KERNEL_CHECK();
}

} // namespace llaisys::ops::iluvatar
