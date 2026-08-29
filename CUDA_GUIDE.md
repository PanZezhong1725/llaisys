# CUDA 编程基础指南

## 1. CUDA 简介

CUDA (Compute Unified Device Architecture) 是 NVIDIA 推出的并行计算平台和编程模型，允许开发者使用 C/C++ 等语言在 GPU 上进行通用计算。

## 2. 核心概念

### 2.1 主机 (Host) 和设备 (Device)
- **主机**: CPU 和主机内存
- **设备**: GPU 和显存

### 2.2 内核 (Kernel)
- 在 GPU 上执行的函数
- 使用 `__global__` 关键字声明
- 由主机代码调用

### 2.3 线程层次结构
- **线程 (Thread)**: 最基本的执行单元
- **线程块 (Block)**: 一组线程，可以共享内存
- **网格 (Grid)**: 一组线程块

## 3. 基本语法

### 3.1 函数声明
```cuda
__global__ void kernel_function() {
    // GPU 代码
}

__host__ void host_function() {
    // CPU 代码
}

__device__ void device_function() {
    // GPU 设备函数，只能被内核调用
}
```

### 3.2 内存管理
```cuda
// 分配显存
float *d_data;
cudaMalloc(&d_data, size * sizeof(float));

// 主机到设备拷贝
cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

// 设备到主机拷贝
cudaMemcpy(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

// 释放显存
cudaFree(d_data);
```

### 3.3 内核调用
```cuda
// 定义线程块和网格大小
dim3 blockSize(256);
dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

// 调用内核
kernel_function<<<gridSize, blockSize>>>(args);
```

## 4. 简单示例

### 4.1 向量加法
```cuda
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000;
    size_t size = n * sizeof(float);
    
    // 主机内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 拷贝数据到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 调用内核
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // 拷贝结果回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // 清理
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
```

## 5. 内存类型

### 5.1 全局内存 (Global Memory)
- 容量大，但访问速度慢
- 所有线程都可以访问

### 5.2 共享内存 (Shared Memory)
- 容量小，但访问速度快
- 同一线程块内的线程共享

### 5.3 常量内存 (Constant Memory)
- 只读，缓存优化
- 所有线程都可以访问

### 5.4 纹理内存 (Texture Memory)
- 只读，针对空间局部性优化

## 6. 性能优化

### 6.1 内存访问模式
- 合并访问 (Coalesced Access)
- 避免 bank conflicts

### 6.2 线程块大小
- 通常选择 128、256、512 等
- 考虑寄存器和共享内存限制

### 6.3 占用率 (Occupancy)
- 活跃线程块数量与最大线程块数量的比率
- 影响性能

## 7. 常用库

### 7.1 cuBLAS
- 线性代数库
- 矩阵乘法、向量运算等

### 7.2 cuDNN
- 深度学习库
- 卷积、池化、RNN 等

### 7.3 Thrust
- 并行算法库
- 类似 STL 的接口

## 8. 调试工具

### 8.1 CUDA-GDB
- 命令行调试器

### 8.2 NVIDIA Nsight
- 图形化调试和性能分析工具

### 8.3 cuda-memcheck
- 内存错误检查工具

## 9. 学习资源

### 9.1 官方文档
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)

### 9.2 教程
- [CUDA C/C++ Basics](https://developer.nvidia.com/blog/cuda-c-c-basics/)
- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

### 9.3 书籍
- 《CUDA C Programming Guide》
- 《Programming Massively Parallel Processors》

## 10. 实践建议

1. **从简单开始**: 先实现简单的向量运算
2. **理解内存模型**: 掌握不同内存类型的特点
3. **性能分析**: 使用工具分析性能瓶颈
4. **参考示例**: 学习 NVIDIA 提供的示例代码
5. **逐步优化**: 从正确性到性能优化

## 11. 与 LLAISYS 的结合

在 Assignment #4 中，你需要：

1. **实现 CUDA Runtime APIs**:
   - 内存分配/释放
   - 内存拷贝
   - 设备管理

2. **实现 CUDA 算子**:
   - 为每个算子编写 CUDA 内核
   - 优化内存访问模式
   - 使用 CUDA 库加速

3. **集成到模型**:
   - 修改模型代码支持 CUDA
   - 管理设备内存
   - 优化推理性能

## 12. 常见问题

### 12.1 内存错误
- 检查内存分配是否成功
- 确保内存拷贝大小正确
- 避免越界访问

### 12.2 性能问题
- 检查内存访问模式
- 优化线程块大小
- 使用性能分析工具

### 12.3 编译错误
- 检查 CUDA 语法
- 确保包含正确的头文件
- 链接正确的库

## 13. 下一步

1. 安装 CUDA Toolkit
2. 编写第一个 CUDA 程序
3. 学习 LLAISYS 的 CUDA 集成
4. 实现 Assignment #4

希望这个指南对你有帮助！
