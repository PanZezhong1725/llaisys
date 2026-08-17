#include "add_musa.hpp"

#include "../../../utils.hpp"
#include "../../../utils/musa_check.hpp"

#include <musa_fp16.h>
#include <musa_bf16.h>

template <typename T>
__global__ void add_kernel(const T *a, const T *b, T *c, size_t n) {
    // 每个线程用这个公式算出自己负责的元素下标（MUSA 与 CUDA 语法完全一致）
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {  // 防越界：数据可能不满一个 block
        // 统一先转成 float 相加再转回，保证和 CPU 端精度一致
        c[i] = (T)((float)a[i] + (float)b[i]);
    }
}

namespace llaisys::ops::musa {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t n) {
    // 启动配置：每块 256 个线程；块数 = ceil(n / 256)，保证能覆盖全部 n 个元素
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b),
            reinterpret_cast<float *>(c), n);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<__half><<<blocks, threads>>>(
            reinterpret_cast<const __half *>(a), reinterpret_cast<const __half *>(b),
            reinterpret_cast<__half *>(c), n);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<__mt_bfloat16><<<blocks, threads>>>(
            reinterpret_cast<const __mt_bfloat16 *>(a), reinterpret_cast<const __mt_bfloat16 *>(b),
            reinterpret_cast<__mt_bfloat16 *>(c), n);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    // 检查启动本身有没有错（比如参数非法）
    CHECK_MUSA(musaGetLastError());
    // 阻塞等待 GPU 算完，再返回给上层
    CHECK_MUSA(musaDeviceSynchronize());
}

} // namespace llaisys::ops::musa
