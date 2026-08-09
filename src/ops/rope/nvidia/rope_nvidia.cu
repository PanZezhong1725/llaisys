#include "rope_nvidia.cuh"

#include "../../nvidia_common.cuh"

#include <cmath>
#include <cstdint>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void ropeKernel(
    T *out, const T *in, const int64_t *positions, float theta,
    size_t seq_len, size_t num_heads, size_t head_dim) {
    const size_t half_dim = head_dim / 2;
    const size_t pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t pair_count = seq_len * num_heads * half_dim;
    if (pair_index >= pair_count) {
        return;
    }
    const size_t pair = pair_index % half_dim;
    const size_t head_index = pair_index / half_dim;
    const size_t head = head_index % num_heads;
    const size_t seq = head_index / num_heads;
    const float exponent =
        2.0f * static_cast<float>(pair) / static_cast<float>(head_dim);
    const float angle =
        static_cast<float>(positions[seq]) / powf(theta, exponent);
    const T sine = fromFloat<T>(sinf(angle));
    const T cosine = fromFloat<T>(cosf(angle));
    const size_t base = (seq * num_heads + head) * head_dim;
    const size_t first = base + pair;
    const size_t second = base + half_dim + pair;
    const float a = toFloat(in[first]);
    const float b = toFloat(in[second]);
    const T a_cos = fromFloat<T>(a * toFloat(cosine));
    const T b_sin = fromFloat<T>(b * toFloat(sine));
    const T b_cos = fromFloat<T>(b * toFloat(cosine));
    const T a_sin = fromFloat<T>(a * toFloat(sine));
    out[first] = fromFloat<T>(toFloat(a_cos) - toFloat(b_sin));
    out[second] = fromFloat<T>(toFloat(b_cos) + toFloat(a_sin));
}

template <typename T>
void launch(std::byte *out, const std::byte *in, const std::byte *pos_ids,
            float theta, size_t seq_len, size_t num_heads, size_t head_dim) {
    const size_t pairs = seq_len * num_heads * (head_dim / 2);
    const int blocks = static_cast<int>((pairs + THREADS - 1) / THREADS);
    ropeKernel<<<blocks, THREADS, 0, currentStream()>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len,
        num_heads, head_dim);
    checkKernelLaunch();
}

} // namespace

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, float theta, size_t seq_len,
          size_t num_heads, size_t head_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, in, pos_ids, theta, seq_len, num_heads,
                             head_dim);
    case LLAISYS_DTYPE_F16:
        return launch<__half>(out, in, pos_ids, theta, seq_len, num_heads,
                              head_dim);
    case LLAISYS_DTYPE_BF16:
        return launch<__nv_bfloat16>(out, in, pos_ids, theta, seq_len,
                                     num_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
