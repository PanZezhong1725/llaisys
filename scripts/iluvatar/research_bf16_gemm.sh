#!/usr/bin/env bash
# 研究用：为什么 linear 的 bf16 在 Iluvatar 上会失败。
# 已经确认 cublasSgemmEx(..., CUDA_R_16BF, ...) 返回 status 15
# (CUBLAS_STATUS_NOT_SUPPORTED)。这一步不改代码，只是挖一下：
#   1. 这台机器的 cublas 头文件里到底有没有 bf16 相关的声明/函数
#      （尤其是更通用的 cublasGemmEx，它和 cublasSgemmEx 是两个不同的函数）。
#   2. cublasGemmEx 的签名长什么样（老版本 cuBLAS 用 cudaDataType 作为
#      computeType，新版本引入了专门的 cublasComputeType_t 枚举——两者
#      不兼容，得先看清楚这台机器头文件里实际是哪种，再决定要不要写测试代码调用）。
#
# 用法: cd 到 llaisys 仓库根目录，
#   bash scripts/iluvatar/research_bf16_gemm.sh
# 把完整输出贴回来。

set -u

echo "===================================================="
echo "[0] 确认 cublas 头文件实际路径和版本"
echo "===================================================="
CUBLAS_H=$(find /usr/local/corex* -iname "cublas_v2.h" 2>/dev/null | head -1)
CUBLAS_API_H=$(find /usr/local/corex* -iname "cublas_api.h" 2>/dev/null | head -1)
echo "cublas_v2.h: $CUBLAS_H"
echo "cublas_api.h: $CUBLAS_API_H"

echo
echo "===================================================="
echo "[1] cublasGemmEx 的声明（更通用的函数，和 cublasSgemmEx 不是一个）"
echo "===================================================="
grep -n -A 20 "cublasGemmEx" "$CUBLAS_API_H" 2>/dev/null | head -60

echo
echo "===================================================="
echo "[2] cublasComputeType_t 是否存在（新枚举，还是老的直接用 cudaDataType）"
echo "===================================================="
grep -n "cublasComputeType_t" "$CUBLAS_API_H" 2>/dev/null | head -20
echo "---"
grep -rn "CUBLAS_COMPUTE_" /usr/local/corex*/include/*.h 2>/dev/null | head -20

echo
echo "===================================================="
echo "[3] CUDA_R_16BF 这个数据类型枚举是否定义、在哪个头文件"
echo "===================================================="
grep -rn "CUDA_R_16BF" /usr/local/corex*/include/*.h 2>/dev/null

echo
echo "===================================================="
echo "[4] 头文件里 bf16/bfloat16 相关的所有提及（找找有没有专门的 bf16 GEMM 接口）"
echo "===================================================="
grep -rln "bfloat16\|BF16" /usr/local/corex*/include/*.h 2>/dev/null

echo
echo "===================================================="
echo "[5] cublasSgemmEx 本身的声明（对照确认我们调用的这个函数的真实签名）"
echo "===================================================="
grep -n -A 20 "cublasSgemmEx" "$CUBLAS_API_H" 2>/dev/null | head -30

echo
echo "===================================================="
echo "全部输出结束，请把上面内容贴回去"
echo "===================================================="
