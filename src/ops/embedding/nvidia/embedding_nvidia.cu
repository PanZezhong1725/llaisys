#include "embedding_nvidia.hpp"

#include "../../../utils.hpp"
#include "../../../utils/cuda_check.hpp"

#include <cfloat>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void embedding_kernel(T *out, const std::int64_t *index, const T *weight, size_t num_rows, size_t num_cols) {
    // 一元素一线程：遍历输出元素总数，还原 (row, col)
    const size_t total = num_rows * num_cols;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += blockDim.x * gridDim.x) {
        size_t row = i / num_cols;
        size_t col = i % num_cols;
        std::int64_t idx = index[row];
        out[i] = weight[idx * num_cols + col];
    }
}

namespace llaisys::ops::nvidia {

void embedding(tensor_t out, tensor_t index, tensor_t weight, llaisysDataType_t type, size_t num_rows, size_t num_cols) {
    // 一元素一线程：每块 256 线程；块数 = ceil(输出元素总数 / 256)
    const int threads = 256;
    const size_t total = num_rows * num_cols;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<float *>(out->data()),
            reinterpret_cast<const std::int64_t *>(index->data()),
            reinterpret_cast<const float *>(weight->data()),
            num_rows, num_cols);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<__half *>(out->data()),
            reinterpret_cast<const std::int64_t *>(index->data()),
            reinterpret_cast<const __half *>(weight->data()),
            num_rows, num_cols);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<__nv_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<__nv_bfloat16 *>(out->data()),
            reinterpret_cast<const std::int64_t *>(index->data()),
            reinterpret_cast<const __nv_bfloat16 *>(weight->data()),
            num_rows, num_cols);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

} // namespace llaisys::ops::nvidia