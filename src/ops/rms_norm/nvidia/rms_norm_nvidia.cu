#include "rms_norm_nvidia.cuh"

#include "../../../device/nvidia/nvidia_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../../../device/nvidia/nvidia_dtype.cuh"

using llaisys::device::nvidia::to_float;
using llaisys::device::nvidia::from_float;

namespace llaisys::ops::nvidia {

namespace {

template <typename T>
__global__ void rms_norm_kernel(
    T *out,
    const T *in,
    const T *weight,
    size_t hidden_size,
    float eps
) {
    // 一个 block 负责一行，hidden = 1536, 每个线程负责 1536 / 256 = 6 个元素
    const size_t row = blockIdx.x;
    const size_t tid = threadIdx.x;
    const T *row_in = in + row * hidden_size;
    T *row_out = out + row * hidden_size;

    // 每个 thread 先计算自己的部分平方和
    float sum_sq = 0.0f;

    for (size_t col = tid; col < hidden_size; col += blockDim.x) {
        float x = to_float<T>(row_in[col]);
        sum_sq += x * x;
    }

    // block reduction
    extern __shared__ float shared[];

    shared[tid] = sum_sq;

    __syncthreads();

    // blockDim.x = 256
    // 256 -> 128 -> 64 -> ... -> 1
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    // shared[0] 现在是这一整行 sum(x^2)
    float inv_rms = rsqrtf(shared[0] / static_cast<float>(hidden_size) + eps);

    // 再并行写 output
    for (size_t col = tid; col < hidden_size; col += blockDim.x) {
        float x = to_float<T>(row_in[col]);
        float w = to_float<T>(weight[col]);
        float y = x * inv_rms * w;
        row_out[col] = from_float<T>(y);
    }
}

}   // namespace

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t seqlen,
    size_t hidden_size,
    float eps,
    llaisysStream_t stream_
) {
    constexpr int BLOCK_SIZE = 256;

    dim3 grid(seqlen);
    dim3 block(BLOCK_SIZE);

    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);

    switch (dtype) {

    case LLAISYS_DTYPE_F32:

        rms_norm_kernel<float>
            <<<grid,
               block,
               shared_mem_size,
               stream>>>(
                reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(in),
                reinterpret_cast<const float *>(weight),
                hidden_size,
                eps
            );

        break;

    case LLAISYS_DTYPE_F16:

        rms_norm_kernel<__half>
            <<<grid,
               block,
               shared_mem_size,
               stream>>>(
                reinterpret_cast<__half *>(out),
                reinterpret_cast<const __half *>(in),
                reinterpret_cast<const __half *>(weight),
                hidden_size,
                eps
            );

        break;

    case LLAISYS_DTYPE_BF16:

        rms_norm_kernel<__nv_bfloat16>
            <<<grid,
               block,
               shared_mem_size,
               stream>>>(
                reinterpret_cast<__nv_bfloat16 *>(out),
                reinterpret_cast<const __nv_bfloat16 *>(in),
                reinterpret_cast<const __nv_bfloat16 *>(weight),
                hidden_size,
                eps
            );

        break;

    default:
        throw std::runtime_error(
            "Unsupported dtype for NVIDIA RMSNorm"
        );
    }

    CUDA_KERNEL_CHECK();
}


} // namespace llaisys::ops::nvidia