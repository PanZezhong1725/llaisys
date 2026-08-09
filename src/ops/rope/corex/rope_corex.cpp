#include "rope_corex.cuh"

#include "../../corex_common.cuh"

#include <cmath>
#include <cstdint>

namespace llaisys::ops::corex {
namespace {

template <typename T>
__global__ void rotatePairs(T *out, const T *in, const int64_t *positions,
                            float theta, size_t sequence_length,
                            size_t head_count, size_t head_width) {
    const size_t half = head_width / 2;
    const size_t pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_pairs = sequence_length * head_count * half;
    if (pair_id >= total_pairs) {
        return;
    }
    const size_t dimension = pair_id % half;
    const size_t vector_id = pair_id / half;
    const size_t sequence = vector_id / head_count;
    const float power =
        2.0f * static_cast<float>(dimension) / static_cast<float>(head_width);
    const float angle =
        static_cast<float>(positions[sequence]) / powf(theta, power);
    const T sine = fromFloat<T>(sinf(angle));
    const T cosine = fromFloat<T>(cosf(angle));
    const size_t base = vector_id * head_width;
    const size_t low = base + dimension;
    const size_t high = low + half;
    const float a = toFloat(in[low]);
    const float b = toFloat(in[high]);
    const T ac = fromFloat<T>(a * toFloat(cosine));
    const T bs = fromFloat<T>(b * toFloat(sine));
    const T bc = fromFloat<T>(b * toFloat(cosine));
    const T as = fromFloat<T>(a * toFloat(sine));
    out[low] = fromFloat<T>(toFloat(ac) - toFloat(bs));
    out[high] = fromFloat<T>(toFloat(bc) + toFloat(as));
}

template <typename T>
void dispatchRope(std::byte *out, const std::byte *in,
                  const std::byte *positions, float theta, size_t seq_len,
                  size_t num_heads, size_t head_dim) {
    const size_t count = seq_len * num_heads * (head_dim / 2);
    const int grid = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    rotatePairs<<<grid, BLOCK_SIZE, 0, currentStream()>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(positions), theta, seq_len,
        num_heads, head_dim);
    checkKernel();
}

} // namespace

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, float theta, size_t seq_len,
          size_t num_heads, size_t head_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return dispatchRope<float>(out, in, pos_ids, theta, seq_len,
                                   num_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return dispatchRope<__half>(out, in, pos_ids, theta, seq_len,
                                    num_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return dispatchRope<__nv_bfloat16>(out, in, pos_ids, theta, seq_len,
                                           num_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::corex
