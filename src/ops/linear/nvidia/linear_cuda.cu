#include "../../../utils.hpp"
#include "linear_cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define CEIL(a, b) (((a) + (b) - 1) / (b))
constexpr int TILE_SIZE = 16;
// 1. Kernel：只负责 GPU 上的具体计算
template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                              size_t M, size_t N, size_t K) {
    // 获取当前的 要计算的 out row 与 col
    __shared__ T ins[TILE_SIZE][TILE_SIZE];
    __shared__ T weights[TILE_SIZE][TILE_SIZE];

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    size_t numTiles =  CEIL(K, TILE_SIZE);
    float sum = 0.0;
    for(int t = 0;t<numTiles; t++){
        int offset = t*TILE_SIZE;
        //load in
        int localcol = offset + threadIdx.x;
        if (row < M && localcol < K){
            ins[threadIdx.y][threadIdx.x] = in[row * K + localcol ];
        }else{
            ins[threadIdx.y][threadIdx.x] = 0;
        }
        //load weight
        int localrow = offset + threadIdx.y;
        if (localrow < K && col < N){
            weights[threadIdx.y][threadIdx.x] = weight[col*K + localrow];
        }else{
            weights[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += (float)(ins[threadIdx.y][k] ) * (float)(weights[k][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        out[row * N + col] = sum;
        // 加上bias
        if (bias != nullptr) {
            out[row * N+ col] += bias[col];
        }
    }
}

// 2. Launcher：负责 grid、block 和 kernel 启动
// in : [M, K], weight : [N, K], bias : [N], out : [M, N]
// M: in 的行数，N: weight 的行数，也就是输出列数，K: 公共维度
// out = in * weight^T + bias
template <typename T>
void launch_linear(T *out, const T *in, const T *weight, const T *bias,
                   size_t M, size_t N, size_t K) {

    dim3 block_size(TILE_SIZE, TILE_SIZE);                  // x 对应列，y 对应行
    dim3 grid_size(CEIL(N, TILE_SIZE), CEIL(M, TILE_SIZE)); // x 覆盖 N，y 覆盖 M

    linear_kernel<<<grid_size, block_size>>>(out, in, weight, bias, M, N, K);
}

// 3. 对外接口：负责 std::byte 转换和 dtype 分发
namespace llaisys::ops::cuda {

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_linear(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            M, N, K);
        return;

    case LLAISYS_DTYPE_BF16:
        launch_linear(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            reinterpret_cast<const __nv_bfloat16 *>(bias),
            M, N, K);
        return;

    case LLAISYS_DTYPE_F16:
        launch_linear(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const __half *>(weight),
            reinterpret_cast<const __half *>(bias),
            M, N, K);
        return;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cuda
