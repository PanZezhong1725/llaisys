#include "nvidia_ops.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace {
using half_t = __half;
using bfloat16_t = __nv_bfloat16;

template <typename T>
__device__ float as_float(T value) { return static_cast<float>(value); }
template <>
__device__ float as_float<half_t>(half_t value) { return __half2float(value); }
template <>
__device__ float as_float<bfloat16_t>(bfloat16_t value) { return __bfloat162float(value); }

template <typename T>
__device__ T from_float(float value) { return static_cast<T>(value); }
template <>
__device__ half_t from_float<half_t>(float value) { return __float2half_rn(value); }
template <>
__device__ bfloat16_t from_float<bfloat16_t>(float value) { return __float2bfloat16(value); }

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}
void check_launch() { check_cuda(cudaGetLastError()); }

template <typename T>
__global__ void add_kernel(T *out, const T *a, const T *b, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = from_float<T>(as_float(a[i]) + as_float(b[i]));
}

template <typename T>
__global__ void argmax_kernel(int64_t *out_idx, T *out_val, const T *values, size_t n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        size_t best = 0;
        float best_value = as_float(values[0]);
        for (size_t i = 1; i < n; ++i) {
            const float candidate = as_float(values[i]);
            const bool candidate_nan = isnan(candidate);
            const bool best_nan = isnan(best_value);
            if ((candidate_nan && !best_nan) || (!candidate_nan && candidate > best_value && !best_nan)) {
                best = i;
                best_value = candidate;
            }
        }
        *out_idx = static_cast<int64_t>(best);
        *out_val = from_float<T>(best_value);
    }
}

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *indices, const T *weight, size_t count, size_t dim) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count * dim) {
        const size_t row = i / dim;
        out[i] = weight[static_cast<size_t>(indices[row]) * dim + i % dim];
    }
}

template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                              size_t batch, size_t input_size, size_t output_size) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * output_size) {
        const size_t row = i / output_size;
        const size_t col = i % output_size;
        float result = bias == nullptr ? 0.0F : as_float(bias[col]);
        for (size_t k = 0; k < input_size; ++k) {
            result += as_float(in[row * input_size + k]) * as_float(weight[col * input_size + k]);
        }
        out[i] = from_float<T>(result);
    }
}

template <typename T>
__global__ void rms_kernel(T *out, const T *in, const T *weight, size_t rows, size_t width, float eps) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0F;
        for (size_t i = 0; i < width; ++i) {
            const float value = as_float(in[row * width + i]);
            sum += value * value;
        }
        const float inv_rms = rsqrtf(sum / static_cast<float>(width) + eps);
        for (size_t i = 0; i < width; ++i) {
            out[row * width + i] = from_float<T>(as_float(in[row * width + i]) * inv_rms * as_float(weight[i]));
        }
    }
}

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *positions,
                            size_t sequence, size_t heads, size_t dim, float theta) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t half = dim / 2;
    if (i < sequence * heads * half) {
        const size_t token = i / (heads * half);
        const size_t head = (i / half) % heads;
        const size_t pair = i % half;
        const size_t base = (token * heads + head) * dim;
        const float exponent = 2.0F * static_cast<float>(pair) / static_cast<float>(dim);
        const float angle = static_cast<float>(positions[token]) / powf(theta, exponent);
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float a = as_float(in[base + pair]);
        const float b = as_float(in[base + half + pair]);
        out[base + pair] = from_float<T>(a * c - b * s);
        out[base + half + pair] = from_float<T>(b * c + a * s);
    }
}

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = as_float(gate[i]);
        out[i] = from_float<T>(as_float(up[i]) * (g / (1.0F + expf(-g))));
    }
}

template <typename T>
__global__ void attention_scores_kernel(float *scores, const T *q, const T *k,
                                        size_t qlen, size_t klen, size_t qheads, size_t kvheads,
                                        size_t dim, float scale) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < qlen * qheads * klen) {
        const size_t key_pos = index % klen;
        const size_t qhead = (index / klen) % qheads;
        const size_t qpos = index / (klen * qheads);
        const size_t last = klen - qlen + qpos;
        if (key_pos > last) {
            scores[index] = -CUDART_INF_F;
            return;
        }
        const size_t kvhead = qhead / (qheads / kvheads);
        const T *query = q + (qpos * qheads + qhead) * dim;
        const T *key = k + (key_pos * kvheads + kvhead) * dim;
        float score = 0.0F;
        for (size_t i = 0; i < dim; ++i) score += as_float(query[i]) * as_float(key[i]);
        scores[index] = score * scale;
    }
}

__global__ void attention_softmax_kernel(float *scores, size_t qlen, size_t klen, size_t qheads) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < qlen * qheads) {
        const size_t qpos = row / qheads;
        const size_t last = klen - qlen + qpos;
        float max_score = -CUDART_INF_F;
        for (size_t key_pos = 0; key_pos <= last; ++key_pos) max_score = fmaxf(max_score, scores[row * klen + key_pos]);
        float denominator = 0.0F;
        for (size_t key_pos = 0; key_pos <= last; ++key_pos) {
            const float value = expf(scores[row * klen + key_pos] - max_score);
            scores[row * klen + key_pos] = value;
            denominator += value;
        }
        for (size_t key_pos = 0; key_pos < klen; ++key_pos) {
            scores[row * klen + key_pos] = key_pos <= last ? scores[row * klen + key_pos] / denominator : 0.0F;
        }
    }
}

template <typename T>
__global__ void attention_output_kernel(T *out, const float *scores, const T *v,
                                        size_t qlen, size_t klen, size_t qheads, size_t kvheads,
                                        size_t value_dim) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < qlen * qheads * value_dim) {
        const size_t dim = index % value_dim;
        const size_t qhead = (index / value_dim) % qheads;
        const size_t qpos = index / (value_dim * qheads);
        const size_t kvhead = qhead / (qheads / kvheads);
        float result = 0.0F;
        for (size_t key_pos = 0; key_pos < klen; ++key_pos) {
            const size_t value_offset = (key_pos * kvheads + kvhead) * value_dim + dim;
            result += scores[(qpos * qheads + qhead) * klen + key_pos] * as_float(v[value_offset]);
        }
        out[index] = from_float<T>(result);
    }
}

template <typename T>
void add_impl(std::byte *out, const std::byte *a, const std::byte *b, size_t n) {
    add_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(a), reinterpret_cast<const T *>(b), n);
    check_launch();
}
template <typename T>
void argmax_impl(std::byte *idx, std::byte *val, const std::byte *input, size_t n) {
    argmax_kernel<<<1, 1>>>(reinterpret_cast<int64_t *>(idx), reinterpret_cast<T *>(val), reinterpret_cast<const T *>(input), n);
    check_launch();
}
template <typename T>
void embedding_impl(std::byte *out, const std::byte *index, const std::byte *weight, size_t count, size_t dim) {
    embedding_kernel<<<static_cast<unsigned>((count * dim + 255) / 256), 256>>>(reinterpret_cast<T *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const T *>(weight), count, dim);
    check_launch();
}
template <typename T>
void linear_impl(std::byte *out, const std::byte *input, const std::byte *weight, const std::byte *bias, size_t batch, size_t input_size, size_t output_size) {
    linear_kernel<<<static_cast<unsigned>((batch * output_size + 255) / 256), 256>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(input), reinterpret_cast<const T *>(weight), reinterpret_cast<const T *>(bias), batch, input_size, output_size);
    check_launch();
}
template <typename T>
void rms_impl(std::byte *out, const std::byte *input, const std::byte *weight, size_t rows, size_t width, float eps) {
    rms_kernel<<<static_cast<unsigned>((rows + 255) / 256), 256>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(input), reinterpret_cast<const T *>(weight), rows, width, eps);
    check_launch();
}
template <typename T>
void rope_impl(std::byte *out, const std::byte *input, const std::byte *positions, size_t sequence, size_t heads, size_t dim, float theta) {
    const size_t n = sequence * heads * dim / 2;
    rope_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(input), reinterpret_cast<const int64_t *>(positions), sequence, heads, dim, theta);
    check_launch();
}
template <typename T>
void swiglu_impl(std::byte *out, const std::byte *gate, const std::byte *up, size_t n) {
    swiglu_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate), reinterpret_cast<const T *>(up), n);
    check_launch();
}
template <typename T>
void attention_impl(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    size_t qlen, size_t klen, size_t qheads, size_t kvheads, size_t dim, size_t value_dim, float scale) {
    float *scores = nullptr;
    const size_t score_count = qlen * qheads * klen;
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&scores), score_count * sizeof(float)));
    attention_scores_kernel<<<static_cast<unsigned>((score_count + 255) / 256), 256>>>(scores, reinterpret_cast<const T *>(q), reinterpret_cast<const T *>(k), qlen, klen, qheads, kvheads, dim, scale);
    check_launch();
    attention_softmax_kernel<<<static_cast<unsigned>((qlen * qheads + 255) / 256), 256>>>(scores, qlen, klen, qheads);
    check_launch();
    const size_t output_count = qlen * qheads * value_dim;
    attention_output_kernel<<<static_cast<unsigned>((output_count + 255) / 256), 256>>>(reinterpret_cast<T *>(out), scores, reinterpret_cast<const T *>(v), qlen, klen, qheads, kvheads, value_dim);
    check_launch();
    check_cuda(cudaFree(scores));
}
} // namespace

namespace llaisys::ops::nvidia {
void add(std::byte *o, const std::byte *a, const std::byte *b, llaisysDataType_t d, size_t n) { switch (d) { case LLAISYS_DTYPE_F32: return add_impl<float>(o,a,b,n); case LLAISYS_DTYPE_F16: return add_impl<half_t>(o,a,b,n); case LLAISYS_DTYPE_BF16: return add_impl<bfloat16_t>(o,a,b,n); default: throw std::invalid_argument("NVIDIA add dtype"); } }
void argmax(std::byte *i, std::byte *v, const std::byte *x, llaisysDataType_t d, size_t n) { switch (d) { case LLAISYS_DTYPE_F32: return argmax_impl<float>(i,v,x,n); case LLAISYS_DTYPE_F16: return argmax_impl<half_t>(i,v,x,n); case LLAISYS_DTYPE_BF16: return argmax_impl<bfloat16_t>(i,v,x,n); default: throw std::invalid_argument("NVIDIA argmax dtype"); } }
void embedding(std::byte *o, const std::byte *i, const std::byte *w, llaisysDataType_t d, size_t n, size_t, size_t dim) { switch (d) { case LLAISYS_DTYPE_F32: return embedding_impl<float>(o,i,w,n,dim); case LLAISYS_DTYPE_F16: return embedding_impl<half_t>(o,i,w,n,dim); case LLAISYS_DTYPE_BF16: return embedding_impl<bfloat16_t>(o,i,w,n,dim); default: throw std::invalid_argument("NVIDIA embedding dtype"); } }
void linear(std::byte *o, const std::byte *i, const std::byte *w, const std::byte *b, llaisysDataType_t d, size_t n, size_t k, size_t m) { switch (d) { case LLAISYS_DTYPE_F32: return linear_impl<float>(o,i,w,b,n,k,m); case LLAISYS_DTYPE_F16: return linear_impl<half_t>(o,i,w,b,n,k,m); case LLAISYS_DTYPE_BF16: return linear_impl<bfloat16_t>(o,i,w,b,n,k,m); default: throw std::invalid_argument("NVIDIA linear dtype"); } }
void rms_norm(std::byte *o, const std::byte *i, const std::byte *w, llaisysDataType_t d, size_t n, size_t k, float e) { switch (d) { case LLAISYS_DTYPE_F32: return rms_impl<float>(o,i,w,n,k,e); case LLAISYS_DTYPE_F16: return rms_impl<half_t>(o,i,w,n,k,e); case LLAISYS_DTYPE_BF16: return rms_impl<bfloat16_t>(o,i,w,n,k,e); default: throw std::invalid_argument("NVIDIA RMS norm dtype"); } }
void rope(std::byte *o, const std::byte *i, const std::byte *p, llaisysDataType_t d, size_t n, size_t h, size_t k, float t) { switch (d) { case LLAISYS_DTYPE_F32: return rope_impl<float>(o,i,p,n,h,k,t); case LLAISYS_DTYPE_F16: return rope_impl<half_t>(o,i,p,n,h,k,t); case LLAISYS_DTYPE_BF16: return rope_impl<bfloat16_t>(o,i,p,n,h,k,t); default: throw std::invalid_argument("NVIDIA RoPE dtype"); } }
void self_attention(std::byte *o, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t d, size_t ql, size_t kl, size_t qh, size_t kh, size_t hd, size_t vd, float s) { switch (d) { case LLAISYS_DTYPE_F32: return attention_impl<float>(o,q,k,v,ql,kl,qh,kh,hd,vd,s); case LLAISYS_DTYPE_F16: return attention_impl<half_t>(o,q,k,v,ql,kl,qh,kh,hd,vd,s); case LLAISYS_DTYPE_BF16: return attention_impl<bfloat16_t>(o,q,k,v,ql,kl,qh,kh,hd,vd,s); default: throw std::invalid_argument("NVIDIA attention dtype"); } }
void swiglu(std::byte *o, const std::byte *g, const std::byte *u, llaisysDataType_t d, size_t n) { switch (d) { case LLAISYS_DTYPE_F32: return swiglu_impl<float>(o,g,u,n); case LLAISYS_DTYPE_F16: return swiglu_impl<half_t>(o,g,u,n); case LLAISYS_DTYPE_BF16: return swiglu_impl<bfloat16_t>(o,g,u,n); default: throw std::invalid_argument("NVIDIA SwiGLU dtype"); } }
} // namespace llaisys::ops::nvidia
