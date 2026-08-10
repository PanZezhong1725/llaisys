// Standalone test to determine the correct cuBLAS call for row-major
//   out = in @ weight^T   (linear)
//   scores = q @ k^T      (attention GEMM1)
//   out = scores @ v      (attention GEMM2)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)
#define CHECK_CUBLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error %d at %d\n", (int)s, __LINE__); exit(1);} } while(0)

// Reference: out[i][j] = sum_k in[i][k] * weight[j][k]  (in @ weight^T)
void ref_linear(float* out, const float* in, const float* weight,
                int seq_len, int in_features, int out_features) {
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < out_features; j++) {
            float s = 0;
            for (int k = 0; k < in_features; k++)
                s += in[i*in_features + k] * weight[j*in_features + k];
            out[i*out_features + j] = s;
        }
}

// Reference: scores[i][j] = sum_k q[i][k] * k[j][k]  (q @ k^T)
void ref_qkt(float* scores, const float* q, const float* k,
             int seq_len, int kv_len, int head_dim) {
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < kv_len; j++) {
            float s = 0;
            for (int hd = 0; hd < head_dim; hd++)
                s += q[i*head_dim + hd] * k[j*head_dim + hd];
            scores[i*kv_len + j] = s;
        }
}

// Reference: out[i][j] = sum_k scores[i][k] * v[k][j]  (scores @ v)
void ref_sv(float* out, const float* scores, const float* v,
            int seq_len, int kv_len, int head_dim) {
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < head_dim; j++) {
            float s = 0;
            for (int k = 0; k < kv_len; k++)
                s += scores[i*kv_len + k] * v[k*head_dim + j];
            out[i*head_dim + j] = s;
        }
}

float maxdiff(const float* a, const float* b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        if (d < 0) d = -d;
        if (d > m) m = d;
    }
    return m;
}

int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // ============ TEST 1: linear out = in @ weight^T ============
    // Correct: (T,N) m=out_features n=seq_len k=in_features lda=k ldb=k ldc=m
    {
        int seq_len = 2, in_features = 4, out_features = 3;
        float in[8] = {0.1f,0.2f,0.3f,0.4f, 0.5f,0.6f,0.7f,0.8f};
        float weight[12] = {1,2,3,4, 5,6,7,8, 9,10,11,12}; // [3,4] row-major
        float out[6] = {0};
        float ref[6];
        ref_linear(ref, in, weight, seq_len, in_features, out_features);

        float *d_in, *d_w, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in, sizeof(in)));
        CHECK_CUDA(cudaMalloc(&d_w, sizeof(weight)));
        CHECK_CUDA(cudaMalloc(&d_out, sizeof(out)));
        CHECK_CUDA(cudaMemcpy(d_in, in, sizeof(in), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_w, weight, sizeof(weight), cudaMemcpyHostToDevice));

        int m = out_features, n = seq_len, k = in_features;
        float alpha = 1.0f, beta = 0.0f;

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, k, &alpha, d_w, CUDA_R_32F, k, d_in, CUDA_R_32F, k,
            &beta, d_out, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CHECK_CUDA(cudaMemcpy(out, d_out, sizeof(out), cudaMemcpyDeviceToHost));
        printf("LINEAR (T,N lda=k ldb=k ldc=m):\n");
        printf("  ref:  [%f %f %f]\n  got:  [%f %f %f]\n", ref[0],ref[1],ref[2], out[0],out[1],out[2]);
        printf("  maxdiff = %f\n", maxdiff(out, ref, 6));

        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_out));
    }

    // ============ TEST 2: attention GEMM1 scores = q @ k^T ============
    // Correct: (T,N) m=kv_len n=seq_len k=head_dim lda=k ldb=k ldc=m
    {
        int seq_len = 2, kv_len = 2, head_dim = 4;
        float q[8] = {0.1f,0.2f,0.3f,0.4f, 0.5f,0.6f,0.7f,0.8f};
        float k[8] = {1,2,3,4, 5,6,7,8};
        float scores[4] = {0};
        float ref[4];
        ref_qkt(ref, q, k, seq_len, kv_len, head_dim);

        float *d_q, *d_k, *d_s;
        CHECK_CUDA(cudaMalloc(&d_q, sizeof(q)));
        CHECK_CUDA(cudaMalloc(&d_k, sizeof(k)));
        CHECK_CUDA(cudaMalloc(&d_s, sizeof(scores)));
        CHECK_CUDA(cudaMemcpy(d_q, q, sizeof(q), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_k, k, sizeof(k), cudaMemcpyHostToDevice));

        int m = kv_len, n = seq_len, kk = head_dim;
        float alpha = 1.0f, beta = 0.0f;

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            m, n, kk, &alpha, d_k, CUDA_R_32F, kk, d_q, CUDA_R_32F, kk,
            &beta, d_s, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CHECK_CUDA(cudaMemcpy(scores, d_s, sizeof(scores), cudaMemcpyDeviceToHost));
        printf("\nGEMM1 (T,N lda=k ldb=k ldc=m):\n");
        printf("  ref:  [%f %f; %f %f]\n", ref[0],ref[1],ref[2],ref[3]);
        printf("  got:  [%f %f; %f %f]\n", scores[0],scores[1],scores[2],scores[3]);
        printf("  maxdiff = %f\n", maxdiff(scores, ref, 4));

        CHECK_CUDA(cudaFree(d_q));
        CHECK_CUDA(cudaFree(d_k));
        CHECK_CUDA(cudaFree(d_s));
    }

    // ============ TEST 3: attention GEMM2 out = scores @ v ============
    // Correct: (N,N) m=head_dim n=seq_len k=kv_len lda=m ldb=k ldc=m
    {
        int seq_len = 2, kv_len = 2, head_dim = 4;
        float scores[4] = {0.5f,0.5f, 0.2f,0.8f};
        float v[8] = {1,2,3,4, 5,6,7,8};
        float out[8] = {0};
        float ref[8];
        ref_sv(ref, scores, v, seq_len, kv_len, head_dim);

        float *d_s, *d_v, *d_o;
        CHECK_CUDA(cudaMalloc(&d_s, sizeof(scores)));
        CHECK_CUDA(cudaMalloc(&d_v, sizeof(v)));
        CHECK_CUDA(cudaMalloc(&d_o, sizeof(out)));
        CHECK_CUDA(cudaMemcpy(d_s, scores, sizeof(scores), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_v, v, sizeof(v), cudaMemcpyHostToDevice));

        int m = head_dim, n = seq_len, k = kv_len;
        float alpha = 1.0f, beta = 0.0f;

        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k, &alpha, d_v, CUDA_R_32F, m, d_s, CUDA_R_32F, k,
            &beta, d_o, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        CHECK_CUDA(cudaMemcpy(out, d_o, sizeof(out), cudaMemcpyDeviceToHost));
        printf("\nGEMM2 (N,N lda=m ldb=k ldc=m):\n");
        printf("  ref:  [%f %f %f %f; %f %f %f %f]\n", ref[0],ref[1],ref[2],ref[3],ref[4],ref[5],ref[6],ref[7]);
        printf("  got:  [%f %f %f %f; %f %f %f %f]\n", out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
        printf("  maxdiff = %f\n", maxdiff(out, ref, 8));

        CHECK_CUDA(cudaFree(d_s));
        CHECK_CUDA(cudaFree(d_v));
        CHECK_CUDA(cudaFree(d_o));
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    printf("\nDONE\n");
    return 0;
}
