#include "self_attention_nvidia.cuh"

#include "../../../device/nvidia/cuda_helpers.cuh"

#include <vector>

namespace llaisys::ops::nvidia {
namespace {

struct ScoreBuffer {
    int device;
    float *data;
    size_t capacity;
};

float *scoreWorkspace(size_t elements) {
    int current = 0;
    device::nvidia::requireCuda(cudaGetDevice(&current), "cudaGetDevice");
    static thread_local std::vector<ScoreBuffer> buffers;
    for (auto &buffer : buffers) {
        if (buffer.device != current) continue;
        if (buffer.capacity < elements) {
            if (buffer.data != nullptr) device::nvidia::requireCuda(cudaFree(buffer.data), "cudaFree attention workspace");
            buffer.data = nullptr;
            device::nvidia::requireCuda(cudaMalloc(reinterpret_cast<void **>(&buffer.data), elements * sizeof(float)), "cudaMalloc attention workspace");
            buffer.capacity = elements;
        }
        return buffer.data;
    }
    ScoreBuffer fresh{current, nullptr, elements};
    device::nvidia::requireCuda(cudaMalloc(reinterpret_cast<void **>(&fresh.data), elements * sizeof(float)), "cudaMalloc attention workspace");
    buffers.push_back(fresh);
    return buffers.back().data;
}

template <class Scalar>
__global__ void computeScores(
    float *scores,
    const Scalar *query,
    const Scalar *key,
    size_t rows,
    size_t query_length,
    size_t key_length,
    size_t query_heads,
    size_t key_value_heads,
    size_t width,
    float scale) {
    const size_t total = rows * key_length;
    const size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t group = query_heads / key_value_heads;
    for (size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; item < total; item += step) {
        const size_t row = item / key_length;
        const size_t source = item - row * key_length;
        const size_t token = row / query_heads;
        const size_t head = row - token * query_heads;
        const size_t allowed = key_length - query_length + token + 1;
        if (source >= allowed) {
            scores[item] = -3.402823466e+38F;
            continue;
        }
        const size_t kv_head = head / group;
        const size_t q_base = (token * query_heads + head) * width;
        const size_t k_base = (source * key_value_heads + kv_head) * width;
        float dot = 0.0f;
        for (size_t component = 0; component < width; ++component) {
            dot += device::nvidia::scalarToFloat(query[q_base + component])
                 * device::nvidia::scalarToFloat(key[k_base + component]);
        }
        scores[item] = dot * scale;
    }
}

__global__ void normalizeScores(float *scores, size_t rows, size_t key_length) {
    __shared__ float scratch[256];
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        float local_max = -3.402823466e+38F;
        for (size_t source = threadIdx.x; source < key_length; source += blockDim.x) {
            local_max = fmaxf(local_max, scores[row * key_length + source]);
        }
        scratch[threadIdx.x] = local_max;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
            if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
            __syncthreads();
        }
        const float maximum = scratch[0];

        float local_sum = 0.0f;
        for (size_t source = threadIdx.x; source < key_length; source += blockDim.x) {
            const size_t offset = row * key_length + source;
            const float probability = expf(scores[offset] - maximum);
            scores[offset] = probability;
            local_sum += probability;
        }
        scratch[threadIdx.x] = local_sum;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1) {
            if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }
        const float inverse_sum = 1.0f / scratch[0];
        for (size_t source = threadIdx.x; source < key_length; source += blockDim.x) {
            scores[row * key_length + source] *= inverse_sum;
        }
        __syncthreads();
    }
}

template <class Scalar>
__global__ void weightedValues(
    Scalar *output,
    const Scalar *value,
    const float *scores,
    size_t rows,
    size_t key_length,
    size_t query_heads,
    size_t key_value_heads,
    size_t value_width) {
    const size_t elements = rows * value_width;
    const size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t group = query_heads / key_value_heads;
    for (size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; item < elements; item += step) {
        const size_t row = item / value_width;
        const size_t component = item - row * value_width;
        const size_t head = row % query_heads;
        const size_t kv_head = head / group;
        float result = 0.0f;
        for (size_t source = 0; source < key_length; ++source) {
            const size_t v_offset = (source * key_value_heads + kv_head) * value_width + component;
            result += scores[row * key_length + source] * device::nvidia::scalarToFloat(value[v_offset]);
        }
        output[item] = device::nvidia::floatToScalar<Scalar>(result);
    }
}

template <class Scalar>
void launch(
    std::byte *output,
    const std::byte *query,
    const std::byte *key,
    const std::byte *value,
    size_t query_length,
    size_t key_length,
    size_t query_heads,
    size_t key_value_heads,
    size_t query_key_width,
    size_t value_width,
    float scale,
    cudaStream_t stream) {
    const size_t rows = query_length * query_heads;
    const size_t score_count = rows * key_length;
    float *scores = scoreWorkspace(score_count);
    constexpr unsigned int block = 256;
    computeScores<<<device::nvidia::gridFor(score_count, block), block, 0, stream>>>(
        scores, reinterpret_cast<const Scalar *>(query), reinterpret_cast<const Scalar *>(key), rows,
        query_length, key_length, query_heads, key_value_heads, query_key_width, scale);
    device::nvidia::requireCuda(cudaGetLastError(), "attention score kernel");

    const unsigned int softmax_grid = static_cast<unsigned int>(rows < 65535 ? rows : 65535);
    normalizeScores<<<softmax_grid, block, 0, stream>>>(scores, rows, key_length);
    device::nvidia::requireCuda(cudaGetLastError(), "attention softmax kernel");

    const size_t output_count = rows * value_width;
    weightedValues<<<device::nvidia::gridFor(output_count, block), block, 0, stream>>>(
        reinterpret_cast<Scalar *>(output), reinterpret_cast<const Scalar *>(value), scores, rows,
        key_length, query_heads, key_value_heads, value_width);
    device::nvidia::requireCuda(cudaGetLastError(), "attention value kernel");
}

} // namespace

void self_attention(std::byte *output, const std::byte *query, const std::byte *key, const std::byte *value, llaisysDataType_t dtype, size_t query_length, size_t key_length, size_t query_heads, size_t key_value_heads, size_t query_key_width, size_t value_width, float scale, llaisysStream_t stream) {
    const cudaStream_t native_stream = device::nvidia::cudaStream(stream);
    switch (dtype) {
    case LLAISYS_DTYPE_F32: return launch<float>(output, query, key, value, query_length, key_length, query_heads, key_value_heads, query_key_width, value_width, scale, native_stream);
    case LLAISYS_DTYPE_F16: return launch<__half>(output, query, key, value, query_length, key_length, query_heads, key_value_heads, query_key_width, value_width, scale, native_stream);
    case LLAISYS_DTYPE_BF16: return launch<__nv_bfloat16>(output, query, key, value, query_length, key_length, query_heads, key_value_heads, query_key_width, value_width, scale, native_stream);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
