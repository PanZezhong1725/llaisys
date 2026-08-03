#include "../../../utils.hpp"
#include "argmax_cuda.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t error = (call);                                              \
        if (error != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); \
            std::abort();                                                        \
        }                                                                        \
    } while (0)

template <typename T>
__global__ void argmax_kernel(int64_t *block_idx, T *block_val, const T *vals, size_t numel, const int64_t *idx_map) {

    // Step 1. 线程局部扫描（thread-local scan）
    T best_val = -INFINITY;;
    int64_t best_idx;
    size_t tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

    for (size_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        if(vals[i] > best_val){
            best_val = vals[i];
            if(idx_map == nullptr){
            best_idx = i;
            }else{
            best_idx = idx_map[i];
            }
        }
    }
    // Step 2. 块内规约（block-level reduction）
    extern __shared__ unsigned char shared_mem[];
    T *shared_val = reinterpret_cast<T *>(shared_mem);
    int64_t *shared_idx = reinterpret_cast<int64_t *>(shared_val + blockDim.x);

    T local_max = -INFINITY;
    // 加载数据
    shared_val[tid] = idx < numel ? best_val : local_max;
    shared_idx[tid] = idx < numel ? static_cast<int64_t>(best_idx) : 0.0;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && shared_val[tid] < shared_val[tid + stride]) {
            shared_val[tid] = shared_val[tid + stride];
            shared_idx[tid] = shared_idx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_idx[blockIdx.x] = shared_idx[0];
        block_val[blockIdx.x] = shared_val[0]; // 每个 block 写出一个局部和
    }
}

// 2. Launcher：负责 grid、block 和 kernel 启动
template <typename T>
void launch_argmax(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);
    size_t shared_bytes = block_size * sizeof(T) + block_size * sizeof(int64_t);
    T *val_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&val_ptr), grid_size * sizeof(T)));
    int64_t *idx_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&idx_ptr), grid_size * sizeof(int64_t)));
    argmax_kernel<<<grid_size, block_size, shared_bytes>>>(idx_ptr,val_ptr, vals, numel,nullptr);
    // 按方案 (b)，这里要变成两轮 kernel launch，而不是现在这一行：
    //   1) 第一轮：grid_size 按 numel 算（下面这行本身没错），但 kernel 不能直接写 max_idx/max_val——
    //      那俩只有 1 个元素的空间，装不下 gridDim.x 个 block 各自的候选结果。要先 cudaMalloc 一块
    //      中间缓冲区（T 类型 grid_size 个 + int64_t 类型 grid_size 个），第一轮写到这块中间缓冲区。
    //   2) 第二轮：<<<1, block_size>>>，输入换成第一轮的中间缓冲区（numel 换成 grid_size），
    //      这次才真正写 max_idx/max_val。跑完记得 cudaFree 中间缓冲区。
    argmax_kernel<<<1, block_size, shared_bytes>>>(max_idx, max_val, val_ptr, grid_size,idx_ptr);
    CHECK_CUDA(cudaFree(val_ptr));
    CHECK_CUDA(cudaFree(idx_ptr));
}

namespace llaisys::ops::cuda {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_argmax(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            numel);

    case LLAISYS_DTYPE_BF16:
        return launch_argmax(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__nv_bfloat16 *>(max_val),
            reinterpret_cast<const __nv_bfloat16 *>(vals),
            numel);

    case LLAISYS_DTYPE_F16:
        return launch_argmax(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__half *>(max_val),
            reinterpret_cast<const __half *>(vals),
            numel);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cuda
