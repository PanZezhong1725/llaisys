#include "linear_nvidia.cuh"

#include "../../nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_resource.cuh"

#include <climits>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t numel,
                              size_t columns) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        out[index] = fromFloat<T>(
            toFloat(out[index]) + toFloat(bias[index % columns]));
    }
}

struct GemmType {
    cudaDataType_t data;
    cublasComputeType_t compute;
};

GemmType gemmType(llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return {CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC};
    case LLAISYS_DTYPE_F16:
        return {CUDA_R_16F, CUBLAS_COMPUTE_32F};
    case LLAISYS_DTYPE_BF16:
        return {CUDA_R_16BF, CUBLAS_COMPUTE_32F};
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
        return {CUDA_R_32F, CUBLAS_COMPUTE_32F};
    }
}

template <typename T>
void launchBias(std::byte *out, const std::byte *bias, size_t m, size_t n) {
    if (bias == nullptr) {
        return;
    }
    const size_t numel = m * n;
    const int blocks = static_cast<int>((numel + THREADS - 1) / THREADS);
    addBiasKernel<<<blocks, THREADS, 0, currentStream()>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(bias),
        numel, n);
    checkKernelLaunch();
}

} // namespace

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type, size_t m,
            size_t n, size_t k) {
    CHECK_ARGUMENT(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
                   "Linear: dimensions exceed cuBLAS integer limits.");
    const auto gemm_type = gemmType(type);
    auto &runtime = llaisys::core::context().runtime();
    auto handle = llaisys::device::nvidia::getResource(
                      runtime.deviceId())
                      .cublas(runtime.stream());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
        &alpha, weight, gemm_type.data, static_cast<int>(k),
        in, gemm_type.data, static_cast<int>(k),
        &beta, out, gemm_type.data, static_cast<int>(n),
        gemm_type.compute, CUBLAS_GEMM_DEFAULT));

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchBias<float>(out, bias, m, n);
    case LLAISYS_DTYPE_F16:
        return launchBias<__half>(out, bias, m, n);
    case LLAISYS_DTYPE_BF16:
        return launchBias<__nv_bfloat16>(out, bias, m, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
