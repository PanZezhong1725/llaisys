#include "argmax_suda.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace llaisys::ops::suda {

// Stage 1: each block reduces its slice and writes (value, index) to the temp buffer.
template <typename T>
__global__ void argmax_block_kernel(const T *vals, size_t size, T *block_vals, int *block_idx, int num_blocks) {
    __shared__ T s_vals[256];
    __shared__ int s_idx[256];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    size_t start = static_cast<size_t>(bid) * blockDim.x;
    size_t end = start + blockDim.x;
    if (end > size) {
        end = size;
    }

    T local_max = static_cast<T>(0);
    int local_idx = 0;
    if (start < size) {
        local_max = vals[start];
        local_idx = static_cast<int>(start);
        for (size_t i = start + 1; i < end; ++i) {
            if (vals[i] > local_max) {
                local_max = vals[i];
                local_idx = static_cast<int>(i);
            }
        }
    }
    s_vals[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[bid] = s_vals[0];
        block_idx[bid] = s_idx[0];
    }
}

// Stage 2: reduce the per-block results.
template <typename T>
__global__ void argmax_final_kernel(const T *block_vals, const int *block_idx, int num_blocks, T *max_val, int *max_idx) {
    T local_max = block_vals[0];
    int local_idx = block_idx[0];
    for (int i = 1; i < num_blocks; ++i) {
        if (block_vals[i] > local_max) {
            local_max = block_vals[i];
            local_idx = block_idx[i];
        }
    }
    if (threadIdx.x == 0) {
        *max_val = local_max;
        *max_idx = local_idx;
    }
}

template <typename T>
static void launch_argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t size) {
    const int block = 256;
    int grid = static_cast<int>((size + block - 1) / block);
    if (grid == 0) {
        grid = 1;
    }

    T *block_vals = nullptr;
    int *block_idx = nullptr;
    cudaMalloc(&block_vals, sizeof(T) * grid);
    cudaMalloc(&block_idx, sizeof(int) * grid);

    argmax_block_kernel<T><<<grid, block>>>(reinterpret_cast<const T *>(vals), size,
                                            block_vals, block_idx, grid);

    argmax_final_kernel<T><<<1, 1>>>(block_vals, block_idx, grid,
                                     reinterpret_cast<T *>(max_val),
                                     reinterpret_cast<int *>(max_idx));

    cudaFree(block_vals);
    cudaFree(block_idx);
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t dtype, size_t size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch_argmax<float>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_F16:
        return launch_argmax<__half>(max_idx, max_val, vals, size);
    case LLAISYS_DTYPE_BF16:
        return launch_argmax<__nv_bfloat16>(max_idx, max_val, vals, size);
    default:
        break;
    }
}

} // namespace llaisys::ops::suda