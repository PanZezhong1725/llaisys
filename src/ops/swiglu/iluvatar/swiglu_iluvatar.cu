#include "../../../utils.hpp"
#include "swiglu_iluvatar.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// swiglu(gate, up) = silu(gate) * up
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// out/gate/up 形状都是 [rows, d]，且都保证 contiguous（op.cpp 里已断言），
// 所以可以整体当成长度 numel = rows*d 的一维数组处理——
// out[idx] 只依赖同一个 idx 上的 gate[idx]/up[idx]，没有跨元素依赖，
// 也不需要任何归约/shared memory，跟 add/embedding 是同一类型的 kernel。
//
// 当前签名还是模板占位符的样子（单个 input + numel）：
// swiglu 实际需要两个输入 gate、up，需要按这个改一下 kernel/launcher 的参数列表。

// 1. Kernel：只负责 GPU 上的具体计算
template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float gate_val = (float)gate[idx];
        gate_val = gate_val / (1 + std::exp(-gate_val)); // silu(gate)
        float up_val = (float)up[idx];
        float swiglu_val = gate_val * up_val;
        out[idx] = swiglu_val;
    }
}

// 2. Launcher：负责 grid、block 和 kernel 启动
template <typename T>
void launch_swiglu(T *out, const T *gate, const T *up,size_t rows,size_t d) {
    constexpr int block_size = 256;
    int numel = rows * d;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    swiglu_kernel<<<grid_size, block_size>>>(out, gate, up, numel);
}

// 3. 对外接口：负责 std::byte 转换和 dtype 分发
namespace llaisys::ops::iluvatar {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t rows, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_swiglu(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            rows,
            d
        );
        return;

    case LLAISYS_DTYPE_BF16:
        launch_swiglu(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(gate),
            reinterpret_cast<const __nv_bfloat16 *>(up),
            rows,
            d
        );
        return;

    case LLAISYS_DTYPE_F16:
        launch_swiglu(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(gate),
            reinterpret_cast<const __half *>(up),
            rows,
            d
        );
        return;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}