# LLAISYS CUDA 集成指南

## 1. 项目结构

在 Assignment #4 中，你需要创建以下文件：

```
src/
├── device/
│   └── nvidia/
│       ├── nvidia_runtime_api.cu
│       └── nvidia_resource.cu
├── ops/
│   ├── add/
│   │   └── nvidia/
│   │       └── add_nvidia.cu
│   ├── argmax/
│   │   └── nvidia/
│   │       └── argmax_nvidia.cu
│   └── ... (其他算子)
└── models/
    └── qwen2_nvidia.cu

xmake/
└── nvidia.lua
```

## 2. 实现步骤

### 2.1 创建 nvidia.lua

```lua
-- xmake/nvidia.lua
target("llaisys-nvidia")
    set_kind("static")
    add_deps("llaisys-core")
    
    -- CUDA 编译选项
    add_cuflags("-gencode arch=compute_70,code=sm_70")
    add_cuflags("-gencode arch=compute_80,code=sm_80")
    
    -- 包含目录
    add_includedirs("$(projectdir)/include")
    add_includedirs("$(projectdir)/src")
    
    -- 源文件
    add_files("$(projectdir)/src/device/nvidia/*.cu")
    add_files("$(projectdir)/src/ops/*/nvidia/*.cu")
    add_files("$(projectdir)/src/models/*_nvidia.cu")
    
    -- 链接库
    add_links("cudart", "cublas", "cudnn")
target_end()
```

### 2.2 实现 CUDA Runtime APIs

```cuda
// src/device/nvidia/nvidia_runtime_api.cu
#include <cuda_runtime.h>
#include "llaisys/runtime.h"

static llaisysResult_t nvidia_malloc(void **ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

static llaisysResult_t nvidia_free(void *ptr) {
    cudaError_t err = cudaFree(ptr);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

static llaisysResult_t nvidia_memcpy_h2d(void *dst, const void *src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

static llaisysResult_t nvidia_memcpy_d2h(void *dst, const void *src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

static llaisysResult_t nvidia_memcpy_d2d(void *dst, const void *src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

static llaisysResult_t nvidia_memset(void *ptr, int value, size_t size) {
    cudaError_t err = cudaMemset(ptr, value, size);
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

static llaisysResult_t nvidia_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}

// 注册 API
extern "C" const LlaisysRuntimeAPI *llaisysGetNvidiaRuntimeAPI() {
    static LlaisysRuntimeAPI api = {
        .malloc = nvidia_malloc,
        .free = nvidia_free,
        .memcpy_h2d = nvidia_memcpy_h2d,
        .memcpy_d2h = nvidia_memcpy_d2h,
        .memcpy_d2d = nvidia_memcpy_d2d,
        .memset = nvidia_memset,
        .synchronize = nvidia_synchronize,
    };
    return &api;
}
```

### 2.3 实现 CUDA 算子示例

```cuda
// src/ops/add/nvidia/add_nvidia.cu
#include <cuda_runtime.h>
#include "llaisys/ops.h"

__global__ void add_kernel(const float *a, const float *b, float *c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" llaisysResult_t llaisysAddNvidia(
    llaisysTensor_t out,
    llaisysTensor_t a,
    llaisysTensor_t b
) {
    size_t n = a->numel;
    const float *a_data = (const float*)a->data;
    const float *b_data = (const float*)b->data;
    float *out_data = (float*)out->data;
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    add_kernel<<<gridSize, blockSize>>>(a_data, b_data, out_data, n);
    
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? LLAISYS_SUCCESS : LLAISYS_ERROR;
}
```

### 2.4 修改模型支持 CUDA

```cuda
// src/models/qwen2_nvidia.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "llaisys/models/qwen2.h"

// 使用 cuBLAS 进行矩阵乘法
__global__ void linear_kernel(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    size_t batch_size,
    size_t in_features,
    size_t out_features
) {
    // 实现线性层
}

extern "C" llaisysResult_t llaisysQwen2NvidiaForward(
    llaisysQwen2Model_t model,
    const int64_t *tokens,
    size_t num_tokens,
    int64_t *output
) {
    // 实现 CUDA 前向传播
    return LLAISYS_SUCCESS;
}
```

## 3. 编译配置

### 3.1 启用 CUDA 支持

```bash
# 配置 xmake
xmake f --nv-gpu=y -cv

# 编译
xmake

# 安装
xmake install
```

### 3.2 测试 CUDA 实现

```bash
# 测试运行时
python test/test_runtime.py --device nvidia

# 测试算子
python test/test_ops.py --device nvidia

# 测试模型
python test/test_infer.py --model [model_path] --device nvidia
```

## 4. 性能优化

### 4.1 内存优化
- 使用合并内存访问
- 避免 bank conflicts
- 合理使用共享内存

### 4.2 计算优化
- 使用 cuBLAS 进行矩阵运算
- 使用 cuDNN 进行深度学习操作
- 优化线程块大小

### 4.3 通信优化
- 减少主机-设备数据传输
- 使用异步操作
- 批量处理数据

## 5. 调试技巧

### 5.1 使用 CUDA-GDB
```bash
cuda-gdb ./your_program
```

### 5.2 使用 Nsight
```bash
nsight-sys ./your_program
```

### 5.3 内存检查
```bash
cuda-memcheck ./your_program
```

## 6. 常见问题

### 6.1 编译错误
- 检查 CUDA 语法
- 确保包含正确的头文件
- 链接正确的库

### 6.2 运行时错误
- 检查内存分配
- 验证内核参数
- 使用调试工具

### 6.3 性能问题
- 分析内存访问模式
- 优化线程块大小
- 使用性能分析工具

## 7. 参考资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Library](https://docs.nvidia.com/cuda/cublas/)
- [cuDNN Library](https://docs.nvidia.com/deeplearning/cudnn/)
- [NVIDIA Nsight Tools](https://developer.nvidia.com/tools)

## 8. 下一步

1. 安装 CUDA Toolkit
2. 创建 nvidia.lua 配置文件
3. 实现 CUDA Runtime APIs
4. 实现 CUDA 算子
5. 集成到模型
6. 测试和优化

希望这个指南对你有帮助！
