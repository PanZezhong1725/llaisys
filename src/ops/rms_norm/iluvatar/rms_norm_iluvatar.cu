#include "../../../utils.hpp"
#include "rms_norm_iluvatar.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// rows: token 数；d: hidden_size。一个 block 处理一行，先规约出 sum-of-squares 再归一化。
template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, float eps, size_t rows, size_t d) {
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    
    float temp = 0.0f;
    for (size_t i = tid; i < d; i+=blockDim.x)
    {   
        size_t idx = row*d + i; 
        temp += (float)in[idx] * (float)in[idx];
    }
    
  
    extern __shared__ float shared[];
    shared[tid] = temp;
    __syncthreads();
    for(size_t stride = blockDim.x/2; stride>0;stride/=2){
        if (tid < stride){
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float rms = sqrtf(shared[0] / static_cast<float>(d) + eps);
    for(size_t i = tid; i < d; i+=blockDim.x){
        out[row * d + i] = (float)weight[i] * (float)in[row * d + i] / rms;
    }
 
    return;
}

template <typename T>
void launch_rms_norm(T *out, const T *in, const T *weight, float eps, size_t rows, size_t d) {
    constexpr int block_size = 256;
    size_t grid_size = rows;
    size_t shared_bytes = block_size * sizeof(float);
    rms_norm_kernel<<<grid_size, block_size,shared_bytes>>>(out, in, weight,eps,rows,d);
}

namespace llaisys::ops::iluvatar {

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t type, size_t rows, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_rms_norm(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            eps,
            rows,
            d);
        return;

    case LLAISYS_DTYPE_BF16:
        launch_rms_norm(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            eps,
            rows,
            d);
        return;

    case LLAISYS_DTYPE_F16:
        launch_rms_norm(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const __half *>(weight),
            eps,
            rows,
            d);
        return;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::iluvatar