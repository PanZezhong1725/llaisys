#include "../../../utils.hpp"
#include "linear_iluvatar.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "../../../device/iluvatar/iluvatar_resource.cuh"
#include <cassert>
#define CEIL(a, b) (((a) + (b) - 1) / (b))
/* 练手用的 tiled-GEMM 手写 kernel，现在 F32/BF16/F16 都已经改用 cublas 计算，
   下面这段暂时不再被调用，留着当参考。
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
*/

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < M * N) out[idx] = out[idx] + bias[idx % N];
}


/* 同样是练手用的 launcher，配合上面注释掉的 linear_kernel，现在没人调用了。
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
*/

// 3. 对外接口：负责 std::byte 转换和 dtype 分发
namespace llaisys::ops::iluvatar {

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K,llaisys::device::DeviceResource *resource) {
    auto handle = static_cast<llaisys::device::iluvatar::Resource*>(resource)->cublasHandle();
    switch (type) {
    case LLAISYS_DTYPE_F32:
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        const auto *in_f32 = reinterpret_cast<const float *>(in);
        const auto *weight_f32 = reinterpret_cast<const float *>(weight);
        const auto *bias_f32 = reinterpret_cast<const float *>(bias);
        auto *out_f32 = reinterpret_cast<float *>(out);

        cublasStatus_t status = cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            &alpha,
            weight_f32,
            static_cast<int>(K),
            in_f32,
            static_cast<int>(K),
            &beta,
            out_f32,
            static_cast<int>(N)
        );
        assert(status == CUBLAS_STATUS_SUCCESS);
        // printf("%d\n", status);
        if(bias != nullptr){
        constexpr int block_size = 256;
        int grid_size = static_cast<int>(CEIL(M * N,block_size));
        add_bias_kernel<<<grid_size, block_size>>>(out_f32, bias_f32 , M, N);
        
        }
        return;
    }
    case LLAISYS_DTYPE_BF16:
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        const auto *in_bf16 =  reinterpret_cast<const __nv_bfloat16*>(in);
        const auto *weight_bf16 = reinterpret_cast<const __nv_bfloat16 *>(weight);
        const auto *bias_bf16 = reinterpret_cast<const __nv_bfloat16 *>(bias);
        auto *out_bf16 = reinterpret_cast<__nv_bfloat16 *>(out);

        cublasStatus_t status = cublasSgemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            &alpha,
            weight_bf16,
            CUDA_R_16BF,
            static_cast<int>(K),
            in_bf16,
            CUDA_R_16BF,
            static_cast<int>(K),
            &beta,
            out_bf16,
            CUDA_R_16BF,
            static_cast<int>(N)
        );
        assert(status == CUBLAS_STATUS_SUCCESS);
        printf("%d\n", status);
        if(bias != nullptr){
        constexpr int block_size = 256;
        int grid_size = static_cast<int>(CEIL(M * N,block_size));

        add_bias_kernel<<<grid_size, block_size>>>(out_bf16, bias_bf16 , M, N);
        
        }
        return;
    }

    case LLAISYS_DTYPE_F16:
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        const auto *in_f16 =  reinterpret_cast<const __half*>(in);
        const auto *weight_f16 = reinterpret_cast<const __half *>(weight);
        const auto *bias_f16 = reinterpret_cast<const __half *>(bias);
        auto *out_f16 = reinterpret_cast<__half *>(out);

        cublasStatus_t status = cublasSgemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            &alpha,
            weight_f16,
            CUDA_R_16F,
            static_cast<int>(K),
            in_f16,
            CUDA_R_16F,
            static_cast<int>(K),
            &beta,
            out_f16,
            CUDA_R_16F,
            static_cast<int>(N)
        );
        assert(status == CUBLAS_STATUS_SUCCESS);
        if(bias != nullptr){
        constexpr int block_size = 256;
        int grid_size = static_cast<int>(CEIL(M * N,block_size));

        add_bias_kernel<<<grid_size, block_size>>>(out_f16, bias_f16 , M, N);
        
        }
        return;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::iluvatar
