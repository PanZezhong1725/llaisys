#include "linear_corex.cuh"

#include "../../corex_common.cuh"
#include "../../../device/corex/corex_resource.cuh"

#include <climits>

namespace llaisys::ops::corex {
namespace {

template <typename T>
__global__ void addRowBias(T *matrix, const T *bias, size_t count,
                           size_t columns) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        matrix[i] = fromFloat<T>(
            toFloat(matrix[i]) + toFloat(bias[i % columns]));
    }
}

struct GemmConfiguration {
    cudaDataType_t input_type;
    cudaDataType_t compute_type;
};

GemmConfiguration selectGemmConfiguration(llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return {CUDA_R_32F, CUDA_R_32F};
    case LLAISYS_DTYPE_F16:
        return {CUDA_R_16F, CUDA_R_32F};
    case LLAISYS_DTYPE_BF16:
        return {CUDA_R_16BF, CUDA_R_32F};
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
        return {CUDA_R_32F, CUDA_R_32F};
    }
}

template <typename T>
void dispatchBias(std::byte *out, const std::byte *bias, size_t rows,
                  size_t columns) {
    if (bias == nullptr) {
        return;
    }
    const size_t count = rows * columns;
    const int grid = static_cast<int>((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    addRowBias<<<grid, BLOCK_SIZE, 0, currentStream()>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(bias), count,
        columns);
    checkKernel();
}

} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type, size_t m,
            size_t n, size_t k) {
    CHECK_ARGUMENT(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
                   "Linear: dimensions exceed CoreX BLAS integer limits.");
    const auto configuration = selectGemmConfiguration(type);
    auto &runtime = llaisys::core::context().runtime();
    auto handle = llaisys::device::corex::getResource(runtime.deviceId())
                      .cublas(runtime.stream());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    COREX_BLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
        &alpha, weight, configuration.input_type, static_cast<int>(k),
        in, configuration.input_type, static_cast<int>(k),
        &beta, out, configuration.input_type, static_cast<int>(n),
        configuration.compute_type, CUBLAS_GEMM_DEFAULT));

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return dispatchBias<float>(out, bias, m, n);
    case LLAISYS_DTYPE_F16:
        return dispatchBias<__half>(out, bias, m, n);
    case LLAISYS_DTYPE_BF16:
        return dispatchBias<__nv_bfloat16>(out, bias, m, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::corex
