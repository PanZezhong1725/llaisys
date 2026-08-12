#include "argmax_iluvatar.cuh"

#include "../../../utils.hpp"
#include "../../../device/iluvatar/iluvatar_dtype.cuh"
#include "../../../device/iluvatar/iluvatar_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace llaisys::ops::iluvatar {

namespace {

using llaisys::device::iluvatar::from_float;
using llaisys::device::iluvatar::to_float;


__device__ inline bool is_better(
    float candidate_val,
    std::int64_t candidate_idx,
    float best_val,
    std::int64_t best_idx
) {
    if (candidate_idx < 0) {
        return false;
    }

    if (best_idx < 0) {
        return true;
    }

    if (candidate_val > best_val) {
        return true;
    }

    if (candidate_val == best_val && candidate_idx < best_idx) {
        return true;
    }

    return false;
}


template <typename T, int BLOCK_SIZE>
__global__ void argmax_kernel(
    std::int64_t *max_idx,
    T *max_val,
    const T *vals,
    size_t numel
) {
    const size_t tid =
        threadIdx.x;

    /*
     * ------------------------------------------------
     * Phase 1:
     * 每个 thread 扫描自己负责的元素，
     * 得到 local maximum。
     * ------------------------------------------------
     */

    float local_max_val = 0.0f;

    /*
     * -1 表示当前 thread
     * 还没有找到有效元素。
     */
    std::int64_t local_max_idx = -1;

    for (size_t i = tid; i < numel; i += BLOCK_SIZE) {
        const float value = to_float<T>(vals[i]);

        const std::int64_t index = static_cast<std::int64_t>(i);

        if (
            is_better(
                value,
                index,
                local_max_val,
                local_max_idx
            )
        ) {
            local_max_val = value;
            local_max_idx = index;
        }
    }


    /*
     * ------------------------------------------------
     * Phase 2:
     * 每个 thread 把自己的 candidate
     * 写入 shared memory。
     * ------------------------------------------------
     */

    __shared__ float shared_val[BLOCK_SIZE];

    __shared__ std::int64_t shared_idx[BLOCK_SIZE];

    shared_val[tid] = local_max_val;

    shared_idx[tid] = local_max_idx;

    __syncthreads();


    /*
     * ------------------------------------------------
     * Phase 3:
     * block reduction
     *
     * 256
     *  ↓
     * 128
     *  ↓
     * 64
     * ...
     *  ↓
     * 1
     * ------------------------------------------------
     */

    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float candidate_val =shared_val[tid + stride];

            const std::int64_t candidate_idx =shared_idx[tid + stride];

            const float current_val = shared_val[tid];

            const std::int64_t current_idx = shared_idx[tid];

            if (
                is_better(
                    candidate_val,
                    candidate_idx,
                    current_val,
                    current_idx
                )
            ) {
                shared_val[tid] = candidate_val;
                shared_idx[tid] = candidate_idx;
            }
        }

        __syncthreads();
    }


    /*
     * ------------------------------------------------
     * Phase 4:
     * thread 0 得到最终结果。
     * ------------------------------------------------
     */

    if (tid == 0) {
        max_idx[0] =shared_idx[0];
        max_val[0] =from_float<T>(shared_val[0]);
    }
}

} // namespace

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t dtype,
    size_t numel,
    llaisysStream_t stream_
) {

    constexpr int BLOCK_SIZE =256;

    const cudaStream_t stream =
        reinterpret_cast<cudaStream_t>(
            stream_
        );

    auto *index =
        reinterpret_cast<
            std::int64_t *
        >(max_idx);


    switch (dtype) {

    case LLAISYS_DTYPE_F32:

        argmax_kernel<
            float,
            BLOCK_SIZE
        >
            <<<1,
               BLOCK_SIZE,
               0,
               stream>>>(
                index,
                reinterpret_cast<
                    float *
                >(max_val),
                reinterpret_cast<
                    const float *
                >(vals),
                numel
            );

        break;


    case LLAISYS_DTYPE_F16:

        argmax_kernel<
            __half,
            BLOCK_SIZE
        >
            <<<1,
               BLOCK_SIZE,
               0,
               stream>>>(
                index,
                reinterpret_cast<
                    __half *
                >(max_val),
                reinterpret_cast<
                    const __half *
                >(vals),
                numel
            );

        break;


    case LLAISYS_DTYPE_BF16:

        argmax_kernel<
            __nv_bfloat16,
            BLOCK_SIZE
        >
            <<<1,
               BLOCK_SIZE,
               0,
               stream>>>(
                index,
                reinterpret_cast<
                    __nv_bfloat16 *
                >(max_val),
                reinterpret_cast<
                    const __nv_bfloat16 *
                >(vals),
                numel
            );

        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    ILUVATAR_CUDA_KERNEL_CHECK();
}

}