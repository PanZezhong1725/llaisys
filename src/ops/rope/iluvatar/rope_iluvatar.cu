#include "rope_iluvatar.cuh"

#include "../../../device/iluvatar/iluvatar_dtype.cuh"
#include "../../../device/iluvatar/iluvatar_utils.cuh"

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace llaisys::ops::iluvatar {

namespace {

using llaisys::device::iluvatar::to_float;
using llaisys::device::iluvatar::from_float;


/*
 * RoPE V1 :
 *  thread = (seq, head, j)
 *  同一个 (seq, j) 下所有 head 的：
 *      powf(theta, exponent)
 *      sinf(phi)
 *      cosf(phi)
 * 完全一样，却被重复计算了 n_heads 次
 *
 */
template <typename T>
__global__ void rope_kernel_v1(
    T *out,
    const T *in,
    const std::int64_t *pos_ids,
    float theta,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim
) {
    const size_t half_dim = head_dim / 2;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = seq_len * n_heads * half_dim;  // 线程总数

    if (idx >= total) {
        return;
    }

    const size_t j = idx % half_dim;
    const size_t tmp = idx / half_dim;
    const size_t head = tmp % n_heads;
    const size_t seq = tmp / n_heads;
    const std::int64_t position = pos_ids[seq];

    if (position < 0) {
        return;
    }

    //  与 CPU 保持相同公式：
    //
    //  exponent = 2*j / head_dim
    //  freq_scale = theta^exponent
    //  phi = position / freq_scale

    const float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
    const float freq_scale = powf(theta, exponent);
    const float phi = static_cast<float>(position) / freq_scale;

    const float cos_phi = cosf(phi);
    const float sin_phi = sinf(phi);

    const size_t base = (seq * n_heads + head) * head_dim;

    const size_t a_idx = base + j;
    const size_t b_idx = base + half_dim + j;

    const float a = to_float<T>(in[a_idx]);
    const float b = to_float<T>(in[b_idx]);

    const float rotated_a = a * cos_phi - b * sin_phi;
    const float rotated_b = b * cos_phi + a * sin_phi;

    out[a_idx] = from_float<T>(rotated_a);
    out[b_idx] = from_float<T>(rotated_b);
}


/*
 * RoPE V2 :
 *  block = (seq, j)    thread = head
 *
 *      blockIdx.x = 0
 *          → seq=0, j=0
 *      blockIdx.x = 1
 *          → seq=0, j=1
 *      ...
 *      blockIdx.x = half_dim
 *          → seq=1, j=0
 *
 * V1 --> V2 ：
 *  发现不同 head 之间存在公共计算，然后改变 thread mapping，让公共计算只执行一次
 *
 * V2 缺点 ：另一个重复计算 exponent  freq_scale，对于固定 j，它与：seq head position 全部无关
 *      每个 seq 都会计算
 */
template <typename T>
__global__ void rope_kernel_v2(
    T *out,
    const T *in,
    const std::int64_t *pos_ids,
    float theta,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim
) {
    const size_t half_dim = head_dim / 2;

    // 一个 block 对应一个 (seq, j)
    const size_t pair_idx = static_cast<size_t>(blockIdx.x);
    const size_t total_pairs = seq_len * half_dim;

    if (pair_idx >= total_pairs) {
        return;
    }

    const size_t j = pair_idx % half_dim;
    const size_t seq = pair_idx / half_dim;

    // 一个 block = 一个 warp = 一个 (seq,j)
    //
    // lane 0
    //  ├── position
    //  ├── pow
    //  └── sincos
    //       │
    //       │ warp shuffle broadcast
    //       ▼
    // lane 0~31
    //       │
    //       └── 各自处理 head

    float cos_phi = 0.0f;
    float sin_phi = 0.0f;

    int valid = 1;

    if (threadIdx.x == 0) {
        const std::int64_t position = pos_ids[seq];

        if (position < 0) {
            valid = 0;
        } else {
            const float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
            const float freq_scale = powf(theta, exponent);
            const float phi = static_cast<float>(position) / freq_scale;

            // 同时计算 sin 和 cos，比分别调用 sinf/cosf 更合适
            sincosf(phi, &sin_phi, &cos_phi);
        }
    }

    valid = __shfl_sync(0xffffffff, valid, 0);
    sin_phi = __shfl_sync(0xffffffff, sin_phi, 0);
    cos_phi = __shfl_sync(0xffffffff, cos_phi, 0);

    if (!valid) {
        return;
    }

    // block 内 threads 并行处理 heads
    for (size_t head = threadIdx.x; head < n_heads; head += blockDim.x) {
        const size_t base = (seq * n_heads + head) * head_dim;

        const size_t a_idx = base + j;
        const size_t b_idx = base + half_dim + j;

        const float a = to_float<T>(in[a_idx]);
        const float b = to_float<T>(in[b_idx]);

        const float rotated_a = a * cos_phi - b * sin_phi;
        const float rotated_b = b * cos_phi + a * sin_phi;

        out[a_idx] = from_float<T>(rotated_a);
        out[b_idx] = from_float<T>(rotated_b);
    }
}

// TODO : RoPE V3 缓存 freq_scale[j] = theta^(2j/head_dim)


} // namespace

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    float theta,
    llaisysDataType_t type,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    llaisysStream_t stream_
) {
    const size_t half_dim = head_dim / 2;

    // V1：一个 CUDA thread 负责一个 (seq, head, j)，一次完成一对元素 j 与 half_dim + j 的旋转
    // constexpr int BLOCK_SIZE = 256;
    // const size_t total = seq_len * n_heads * half_dim;
    //
    // if (total == 0) {
    //     return;
    // }
    //
    // const int grid_size =static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // V2
    constexpr int BLOCK_SIZE = 32;
    const size_t grid_size = seq_len * half_dim;

    if (grid_size == 0) {
        return;
    }


    const cudaStream_t stream =reinterpret_cast<cudaStream_t>(stream_);

    const auto *positions = reinterpret_cast<const std::int64_t *>(pos_ids);

    switch (type) {

    case LLAISYS_DTYPE_F32:

        rope_kernel_v2<float>
            <<<grid_size,
               BLOCK_SIZE,
               0,
               stream>>>(
                reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(in),
                positions,
                theta,
                seq_len,
                n_heads,
                head_dim
            );

        break;

    case LLAISYS_DTYPE_F16:

        rope_kernel_v2<__half>
            <<<grid_size,
               BLOCK_SIZE,
               0,
               stream>>>(
                reinterpret_cast<__half *>(out),
                reinterpret_cast<const __half *>(in),
                positions,
                theta,
                seq_len,
                n_heads,
                head_dim
            );

        break;

    case LLAISYS_DTYPE_BF16:

        rope_kernel_v2<__nv_bfloat16>
            <<<grid_size,
               BLOCK_SIZE,
               0,
               stream>>>(
                reinterpret_cast<__nv_bfloat16 *>(out),
                reinterpret_cast<const __nv_bfloat16 *>(in),
                positions,
                theta,
                seq_len,
                n_heads,
                head_dim
            );

        break;

    default:
        throw std::runtime_error(
            "RoPE: unsupported ILUVATAR datatype."
        );
    }

    ILUVATAR_CUDA_KERNEL_CHECK();
}

} // namespace llaisys::ops::iluvatar
