#include "../../../utils.hpp"
#include "rope_cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// 手写 kernel（cuDNN RoPE Graph 节点只支持 f16/bf16 不支持 f32，探索后放弃，见 CLAUDE.md）。
template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, float theta,
                             size_t nhead, size_t d) {
    size_t i = blockIdx.x;  // token 位置
    size_t h = blockIdx.y;  // head 编号
    size_t tid = threadIdx.x;
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

template <typename T>
void launch_rope(T *out, const T *in, const int64_t *pos_ids, float theta,
                  size_t seqlen, size_t nhead, size_t d) {
    constexpr int block_size = 256;
    dim3 grid(static_cast<unsigned int>(seqlen), static_cast<unsigned int>(nhead));

    rope_kernel<<<grid, block_size>>>(out, in, pos_ids, theta, nhead, d);
}

namespace llaisys::ops::cuda {

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

} // namespace llaisys::ops::cuda
