#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace llaisys::ops::cpu {

// Helper: convert element to float based on dtype
static inline float to_float(const std::byte *ptr, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return *reinterpret_cast<const float *>(ptr);
    case LLAISYS_DTYPE_F16:
        return llaisys::utils::cast<float>(*reinterpret_cast<const llaisys::fp16_t *>(ptr));
    case LLAISYS_DTYPE_BF16:
        return llaisys::utils::cast<float>(*reinterpret_cast<const llaisys::bf16_t *>(ptr));
    default:
        return 0.0f;
    }
}

// Helper: write float back to dtype
static inline void from_float(float val, std::byte *ptr, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        *reinterpret_cast<float *>(ptr) = val;
        break;
    case LLAISYS_DTYPE_F16:
        *reinterpret_cast<llaisys::fp16_t *>(ptr) = llaisys::utils::cast<llaisys::fp16_t>(val);
        break;
    case LLAISYS_DTYPE_BF16:
        *reinterpret_cast<llaisys::bf16_t *>(ptr) = llaisys::utils::cast<llaisys::bf16_t>(val);
        break;
    default:
        break;
    }
}

// Softmax over last dimension of a 2D matrix [rows, cols]
static void softmax(std::vector<float> &mat, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        // Find max for numerical stability
        float max_val = mat[r * cols];
        for (size_t c = 1; c < cols; c++) {
            if (mat[r * cols + c] > max_val)
                max_val = mat[r * cols + c];
        }
        // Compute exp and sum
        float sum = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            mat[r * cols + c] = std::exp(mat[r * cols + c] - max_val);
            sum += mat[r * cols + c];
        }
        // Normalize
        for (size_t c = 0; c < cols; c++) {
            mat[r * cols + c] /= sum;
        }
    }
}

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t dtype, size_t seq_len, size_t kv_len, size_t num_heads, size_t head_dim,
                    float scale) {
    // For each head:
    //   attn = softmax(Q * K^T * scale + causal_mask) * V
    // Q: [seq_len, num_heads, head_dim]
    // K, V: [kv_len, num_heads, head_dim] (already repeated for GQA)
    // attn: [seq_len, kv_len]
    // result: [seq_len, head_dim]

    size_t elem_size = 0;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        elem_size = 4;
        break;
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        elem_size = 2;
        break;
    default:
        return;
    }

    // Tensor layout: [seq_len, num_heads, head_dim]
    // Element (i, h, k) is at offset (i * num_heads * head_dim + h * head_dim + k) * elem_size
    size_t q_stride = num_heads * head_dim;  // stride for seq_len dimension in Q
    size_t kv_stride = num_heads * head_dim; // stride for seq_len dimension in K/V (after expansion)

    // Temporary buffers for float computation
    std::vector<float> q_f(seq_len * head_dim);
    std::vector<float> k_f(kv_len * head_dim);
    std::vector<float> v_f(kv_len * head_dim);

    for (size_t h = 0; h < num_heads; h++) {
        // Copy Q for this head into float buffer
        // Q[i][h][k] is at (i * q_stride + h * head_dim + k) * elem_size
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t kd = 0; kd < head_dim; kd++) {
                size_t byte_offset = (i * q_stride + h * head_dim + kd) * elem_size;
                q_f[i * head_dim + kd] = to_float(q + byte_offset, dtype);
            }
        }

        // Copy K, V for this head into float buffers
        // K[i][h][k] is at (i * kv_stride + h * head_dim + k) * elem_size
        for (size_t i = 0; i < kv_len; i++) {
            for (size_t kd = 0; kd < head_dim; kd++) {
                size_t byte_offset = (i * kv_stride + h * head_dim + kd) * elem_size;
                k_f[i * head_dim + kd] = to_float(k + byte_offset, dtype);
                v_f[i * head_dim + kd] = to_float(v + byte_offset, dtype);
            }
        }

        // Compute attention scores: score[i][j] = sum_k Q[i][k] * K[j][k] * scale
        std::vector<float> score(seq_len * kv_len, 0.0f);
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < kv_len; j++) {
                float s = 0.0f;
                for (size_t kd = 0; kd < head_dim; kd++) {
                    s += q_f[i * head_dim + kd] * k_f[j * head_dim + kd];
                }
                score[i * kv_len + j] = s * scale;
            }
        }

        // Apply causal mask: position i can only attend to j <= i + (kv_len - seq_len)
        // This is equivalent to torch's tril(diagonal=kv_len - seq_len)
        // For kv_len == seq_len: j <= i (strict causal)
        // For kv_len > seq_len: j <= i + (kv_len - seq_len) (prefix attention)
        size_t diag_shift = kv_len - seq_len;
        for (size_t i = 0; i < seq_len; i++) {
            size_t max_j = i + diag_shift;
            for (size_t j = max_j + 1; j < kv_len; j++) {
                score[i * kv_len + j] = -std::numeric_limits<float>::infinity();
            }
        }

        // Softmax over last dimension (j)
        softmax(score, seq_len, kv_len);

        // Compute output: out[i][k] = sum_j score[i][j] * V[j][k]
        std::vector<float> out_f(seq_len * head_dim, 0.0f);
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t kd = 0; kd < head_dim; kd++) {
                float val = 0.0f;
                for (size_t j = 0; j < kv_len; j++) {
                    val += score[i * kv_len + j] * v_f[j * head_dim + kd];
                }
                out_f[i * head_dim + kd] = val;
            }
        }

        // Write back to output tensor
        // out[i][h][k] is at (i * q_stride + h * head_dim + k) * elem_size
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t kd = 0; kd < head_dim; kd++) {
                size_t byte_offset = (i * q_stride + h * head_dim + kd) * elem_size;
                from_float(out_f[i * head_dim + kd], out + byte_offset, dtype);
            }
        }
    }
}

} // namespace llaisys::ops::cpu
