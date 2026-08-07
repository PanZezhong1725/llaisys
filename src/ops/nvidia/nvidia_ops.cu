#include "nvidia_ops.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
template <typename T> __device__ float to_float(T x) { return static_cast<float>(x); }
template <typename T> __device__ T from_float(float x) { return static_cast<T>(x); }
template <> __device__ float to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __half from_float<__half>(float x) { return __float2half(x); }
template <> __device__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <> __device__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

template <typename T> __global__ void add_kernel(T *c, const T *a, const T *b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = from_float<T>(to_float(a[i]) + to_float(b[i]));
}
template <typename T> __global__ void argmax_kernel(int64_t *idx, T *val, const T *x, size_t n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float best = to_float(x[0]); size_t bi = 0;
        for (size_t i = 1; i < n; i++) { float v = to_float(x[i]); if (v > best) { best = v; bi = i; } }
        *idx = static_cast<int64_t>(bi); *val = from_float<T>(best);
    }
}
template <typename T> __global__ void embedding_kernel(T *out, const int64_t *idx, const T *w, size_t rows, size_t d, size_t vocab) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * d) { size_t r = i / d; size_t c = i % d; out[i] = w[idx[r] * d + c]; }
    (void)vocab;
}
template <typename T> __global__ void linear_kernel(T *out, const T *in, const T *w, const T *bias, size_t m, size_t k, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m * n) { size_t row = i / n, col = i % n; float sum = bias ? to_float(bias[col]) : 0.0f; for (size_t p = 0; p < k; p++) sum += to_float(in[row * k + p]) * to_float(w[col * k + p]); out[i] = from_float<T>(sum); }
}
template <typename T> __global__ void rms_sum_kernel(float *sums, const T *x, size_t rows, size_t d) {
    size_t row = blockIdx.x; if (threadIdx.x == 0) { float s = 0.0f; for (size_t i = 0; i < d; i++) { float v = to_float(x[row * d + i]); s += v * v; } sums[row] = s / static_cast<float>(d); } (void)rows;
}
template <typename T> __global__ void rms_apply_kernel(T *out, const T *x, const T *w, const float *sums, size_t n, size_t d, float eps) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { size_t c = i % d; float normalized = to_float(from_float<T>(to_float(x[i]) * rsqrtf(sums[i / d] + eps))); out[i] = from_float<T>(normalized * to_float(w[c])); }
}
template <typename T> __global__ void rope_kernel(T *out, const T *x, const int64_t *pos, size_t n, size_t heads, size_t d, float theta) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n * heads * d) { size_t j = i % d, half = d / 2; size_t token = i / (heads * d), base = (i / d) * d; size_t pair = j < half ? j : j - half; double angle = static_cast<double>(pos[token]) / pow(static_cast<double>(theta), 2.0 * static_cast<double>(pair) / static_cast<double>(d)); float c = to_float(from_float<T>(static_cast<float>(cos(angle)))), s = to_float(from_float<T>(static_cast<float>(sin(angle)))); float a = to_float(x[base + pair]), b = to_float(x[base + half + pair]); out[i] = from_float<T>(j < half ? a * c - b * s : b * c + a * s); }
}
template <typename T> __global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float g = to_float(gate[i]); float denominator = 1.0f + to_float(from_float<T>(expf(-g))); float activated = to_float(from_float<T>(g / denominator)); out[i] = from_float<T>(to_float(up[i]) * activated); }
}
template <typename T> __global__ void attention_kernel(T *out, const T *q, const T *k, const T *v, size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t d, size_t dv, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= qlen * nh * dv) return; size_t p = i % dv, h = (i / dv) % nh, qi = i / (nh * dv), kvh = h / (nh / nkvh), maxk = kvlen - qlen + qi, qb = (qi * nh + h) * d; float maxs = -INFINITY;
    for (size_t kj = 0; kj <= maxk; kj++) { size_t kb = (kj * nkvh + kvh) * d; float s = 0.0f; for (size_t z = 0; z < d; z++) s += to_float(q[qb + z]) * to_float(k[kb + z]); s *= scale; if (s > maxs) maxs = s; }
    float den = 0.0f, ans = 0.0f; for (size_t kj = 0; kj <= maxk; kj++) { size_t kb = (kj * nkvh + kvh) * d; float s = 0.0f; for (size_t z = 0; z < d; z++) s += to_float(q[qb + z]) * to_float(k[kb + z]); float a = expf(s * scale - maxs); den += a; ans += to_float(from_float<T>(a)) * to_float(v[(kj * nkvh + kvh) * dv + p]); } out[i] = from_float<T>(ans / den);
}

inline void launch_check() { auto e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template <typename T> void add_t(std::byte *c, const std::byte *a, const std::byte *b, size_t n) { add_kernel<<<(n + 255) / 256, 256>>>((T *)c, (const T *)a, (const T *)b, n); launch_check(); }
template <typename T> void argmax_t(std::byte *i, std::byte *v, const std::byte *x, size_t n) { argmax_kernel<<<1, 1>>>((int64_t *)i, (T *)v, (const T *)x, n); launch_check(); }
template <typename T> void embedding_t(std::byte *o, const std::byte *i, const std::byte *w, size_t r, size_t d, size_t vocab) { embedding_kernel<<<(r * d + 255) / 256, 256>>>((T *)o, (const int64_t *)i, (const T *)w, r, d, vocab); launch_check(); }
template <typename T> void linear_t(std::byte *o, const std::byte *i, const std::byte *w, const std::byte *b, size_t m, size_t k, size_t n) { linear_kernel<<<(m * n + 255) / 256, 256>>>((T *)o, (const T *)i, (const T *)w, (const T *)b, m, k, n); launch_check(); }
template <typename T> void rms_t(std::byte *o, const std::byte *i, const std::byte *w, size_t rows, size_t d, float eps) { float *s = nullptr; cudaMalloc(&s, rows * sizeof(float)); rms_sum_kernel<<<rows, 1>>>(s, (const T *)i, rows, d); rms_apply_kernel<<<(rows * d + 255) / 256, 256>>>((T *)o, (const T *)i, (const T *)w, s, rows * d, d, eps); cudaFree(s); launch_check(); }
template <typename T> void rope_t(std::byte *o, const std::byte *i, const std::byte *p, size_t seq, size_t h, size_t d, float theta) { rope_kernel<<<(seq * h * d + 255) / 256, 256>>>((T *)o, (const T *)i, (const int64_t *)p, seq, h, d, theta); launch_check(); }
template <typename T> void attention_t(std::byte *o, const std::byte *q, const std::byte *k, const std::byte *v, size_t ql, size_t kl, size_t nh, size_t nk, size_t d, size_t dv, float scale) { attention_kernel<<<(ql * nh * dv + 255) / 256, 256>>>((T *)o, (const T *)q, (const T *)k, (const T *)v, ql, kl, nh, nk, d, dv, scale); launch_check(); }
template <typename T> void swiglu_t(std::byte *o, const std::byte *g, const std::byte *u, size_t n) { swiglu_kernel<<<(n + 255) / 256, 256>>>((T *)o, (const T *)g, (const T *)u, n); launch_check(); }

#define DISPATCH3(fn, ...) do { switch (dtype) { case LLAISYS_DTYPE_F32: fn<float>(__VA_ARGS__); break; case LLAISYS_DTYPE_F16: fn<__half>(__VA_ARGS__); break; case LLAISYS_DTYPE_BF16: fn<__nv_bfloat16>(__VA_ARGS__); break; default: throw std::invalid_argument("unsupported CUDA dtype"); } } while (0)
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t dtype, size_t n) { DISPATCH3(add_t, c, a, b, n); }
void argmax(std::byte *i, std::byte *v, const std::byte *x, llaisysDataType_t dtype, size_t n) { DISPATCH3(argmax_t, i, v, x, n); }
void embedding(std::byte *o, const std::byte *i, const std::byte *w, llaisysDataType_t dtype, size_t r, size_t d, size_t vocab) { DISPATCH3(embedding_t, o, i, w, r, d, vocab); }
void linear(std::byte *o, const std::byte *i, const std::byte *w, const std::byte *b, llaisysDataType_t dtype, size_t m, size_t k, size_t n) { DISPATCH3(linear_t, o, i, w, b, m, k, n); }
void rms_norm(std::byte *o, const std::byte *i, const std::byte *w, llaisysDataType_t dtype, size_t r, size_t d, float e) { DISPATCH3(rms_t, o, i, w, r, d, e); }
void rope(std::byte *o, const std::byte *i, const std::byte *p, llaisysDataType_t dtype, size_t s, size_t h, size_t d, float t) { DISPATCH3(rope_t, o, i, p, s, h, d, t); }
void self_attention(std::byte *o, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t dtype, size_t ql, size_t kl, size_t nh, size_t nk, size_t d, size_t dv, float s) { DISPATCH3(attention_t, o, q, k, v, ql, kl, nh, nk, d, dv, s); }
void swiglu(std::byte *o, const std::byte *g, const std::byte *u, llaisysDataType_t dtype, size_t n) { DISPATCH3(swiglu_t, o, g, u, n); }
}
