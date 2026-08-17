#pragma once
#include "../utils.hpp"

namespace llaisys::ops::cpu {

// 向量点积
template <typename T>
inline float dot_product(const T *a, const T *b, size_t n) {
    float acc = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        acc += llaisys::utils::cast<float>(a[i]) * llaisys::utils::cast<float>(b[i]);
    }
    return acc;
}

// 矩阵乘法
// A 的步长为 lda, B 的步长为 ldb, C 的步长为 ldc
template <typename T>
void matmul(T *c, const T *a, const T *b,
            size_t M, size_t N, size_t K,
            size_t lda, size_t ldb, size_t ldc) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            c[i * ldc + j] = llaisys::utils::cast<T>(
                dot_product(a + i * lda, b + j * ldb, K));
        }
    }
}

} // namespace llaisys::ops
