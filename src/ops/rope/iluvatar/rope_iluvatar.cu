#include "../../../utils.hpp"
#include "rope_iluvatar.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// V1：手写 kernel。cuDNN Graph API 的 RoPE 节点（9.24+ 才有）探索过一版，但只支持
// f16/bf16（不支持 f32，官方文档：https://docs.nvidia.com/deeplearning/cudnn/latest/operations/RoPE.html），
// 需要 f32 时 fallback 回这版 + 额外算一个 FREQS 角度张量，先放一放，退回这版手写实现。
// 1. Kernel：只负责 GPU 上的具体计算
template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, float theta,
                             size_t nhead, size_t d) {
    size_t i = blockIdx.x;  // token 位置, [0, seqlen)
    size_t h = blockIdx.y;  // head 编号, [0, nhead)
    size_t tid = threadIdx.x; // 线程编号, [0, blockDim.x)
    size_t base = i * nhead * d + h * d;
    size_t stride = blockDim.x;
    for(size_t j = tid; j< d/2;j+=stride){
        float phi = pos_ids[i] / pow(theta, 2.0f * j/d);
        float cos_phi = cos(phi);
        float sin_phi = sin(phi);
        float a = (float)in[base + j];
        float b = (float)in[base + j + d/2];
        out[base + j] = (float) (a * cos_phi - b * sin_phi);
        out[base + j + d/2] = (float) (b * cos_phi + a * sin_phi);
    }
}

// 2. Launcher：负责 grid、block 和 kernel 启动
template <typename T>
void launch_rope(T *out, const T *in, const int64_t *pos_ids, float theta,
                  size_t seqlen, size_t nhead, size_t d) {
    constexpr int block_size = 256;
    dim3 grid(static_cast<unsigned int>(seqlen), static_cast<unsigned int>(nhead));

    rope_kernel<<<grid, block_size>>>(out, in, pos_ids, theta, nhead, d);
}

// 3. 对外接口：负责 std::byte 转换和 dtype 分发
namespace llaisys::ops::iluvatar {

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d, float theta) {
    const int64_t *pos = reinterpret_cast<const int64_t *>(pos_ids);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_rope(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            pos, theta, seqlen, nhead, d
        );
        return;

    case LLAISYS_DTYPE_BF16:
        launch_rope(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            pos, theta, seqlen, nhead, d
        );
        return;

    case LLAISYS_DTYPE_F16:
        launch_rope(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            pos, theta, seqlen, nhead, d
        );
        return;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::iluvatar
