#include "linear_nvidia.cuh"

#include "../../../device/nvidia/cuda_helpers.cuh"

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace {

void requireCublas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " + std::to_string(static_cast<int>(status)));
    }
}

cublasHandle_t handleForCurrentDevice(cudaStream_t stream) {
    thread_local cublasHandle_t handle = nullptr;
    thread_local int owner = -1;
    int current = 0;
    device::nvidia::requireCuda(cudaGetDevice(&current), "cudaGetDevice");
    if (handle == nullptr || owner != current) {
        cublasHandle_t fresh = nullptr;
        requireCublas(cublasCreate(&fresh), "cublasCreate");
        handle = fresh;
        owner = current;
    }
    requireCublas(cublasSetStream(handle, stream), "cublasSetStream");
    return handle;
}

template <class Scalar>
__global__ void broadcastBias(Scalar *output, const Scalar *bias, size_t elements, size_t columns) {
    const size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < elements; i += step) {
        output[i] = bias[i % columns];
    }
}

template <class Scalar>
void initializeWithBias(std::byte *output, const std::byte *bias, size_t elements, size_t columns, cudaStream_t stream) {
    if (elements == 0) return;
    constexpr unsigned int block = 256;
    broadcastBias<<<device::nvidia::gridFor(elements, block), block, 0, stream>>>(
        reinterpret_cast<Scalar *>(output), reinterpret_cast<const Scalar *>(bias), elements, columns);
    device::nvidia::requireCuda(cudaGetLastError(), "linear bias kernel");
}

cudaDataType_t cudaType(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return CUDA_R_32F;
    case LLAISYS_DTYPE_F16: return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16: return CUDA_R_16BF;
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void fillBias(std::byte *output, const std::byte *bias, llaisysDataType_t dtype, size_t elements, size_t columns, cudaStream_t stream) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return initializeWithBias<float>(output, bias, elements, columns, stream);
    case LLAISYS_DTYPE_F16: return initializeWithBias<__half>(output, bias, elements, columns, stream);
    case LLAISYS_DTYPE_BF16: return initializeWithBias<__nv_bfloat16>(output, bias, elements, columns, stream);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

void linear(
    std::byte *output,
    const std::byte *input,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t dtype,
    size_t rows,
    size_t columns,
    size_t reduction,
    llaisysStream_t stream) {
    const cudaStream_t native_stream = device::nvidia::cudaStream(stream);
    const size_t elements = rows * columns;
    if (elements == 0) return;

    if (bias != nullptr) {
        fillBias(output, bias, dtype, elements, columns, native_stream);
    } else if (reduction == 0) {
        device::nvidia::requireCuda(cudaMemsetAsync(output, 0, elements * utils::dsize(dtype), native_stream), "cudaMemsetAsync");
    }
    if (reduction == 0) return;

    const float alpha = 1.0f;
    const float beta = bias == nullptr ? 0.0f : 1.0f;
    const cudaDataType_t data_type = cudaType(dtype);
    const int m = static_cast<int>(columns);
    const int n = static_cast<int>(rows);
    const int k = static_cast<int>(reduction);

    requireCublas(
        cublasGemmEx(
            handleForCurrentDevice(native_stream),
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &alpha,
            weight,
            data_type,
            k,
            input,
            data_type,
            k,
            &beta,
            output,
            data_type,
            m,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx");
}

} // namespace llaisys::ops::nvidia
