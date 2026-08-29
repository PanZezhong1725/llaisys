# CUDA 代码框架说明

## 已创建的文件结构

```
src/
├── device/
│   └── nvidia/
│       ├── nvidia_runtime_api.cu    # CUDA Runtime API 实现
│       └── nvidia_resource.cu       # CUDA 资源管理
├── ops/
│   ├── add/nvidia/
│   │   └── add_nvidia.cu           # 加法算子 CUDA 实现
│   ├── argmax/nvidia/
│   │   └── argmax_nvidia.cu        # Argmax 算子 CUDA 实现
│   ├── embedding/nvidia/
│   │   └── embedding_nvidia.cu     # Embedding 算子 CUDA 实现
│   ├── linear/nvidia/
│   │   └── linear_nvidia.cu        # 线性层算子 CUDA 实现
│   ├── rearrange/nvidia/
│   │   └── rearrange_nvidia.cu     # 重排算子 CUDA 实现
│   ├── rms_norm/nvidia/
│   │   └── rms_norm_nvidia.cu      # RMS Norm 算子 CUDA 实现
│   ├── rope/nvidia/
│   │   └── rope_nvidia.cu          # RoPE 算子 CUDA 实现
│   ├── self_attention/nvidia/
│   │   └── self_attention_nvidia.cu # 自注意力算子 CUDA 实现
│   └── swiglu/nvidia/
│       └── swiglu_nvidia.cu        # SwiGLU 算子 CUDA 实现
└── xmake/
    └── nvidia.lua                   # NVIDIA CUDA 编译配置
```

## 主要功能

### 1. CUDA Runtime API (nvidia_runtime_api.cu)
- 内存管理：malloc, free, memset
- 内存拷贝：H2D, D2H, D2D
- 设备管理：set_device, get_device, get_device_count
- 同步：synchronize

### 2. CUDA 资源管理 (nvidia_resource.cu)
- cuBLAS 句柄管理
- cuDNN 句柄管理
- CUDA 流管理

### 3. CUDA 算子实现
- **Add**: 元素级加法
- **Argmax**: 求最大值和索引
- **Embedding**: 嵌入查找
- **Linear**: 线性变换（使用 cuBLAS）
- **Rearrange**: 张量重排
- **RMS Norm**: RMS 归一化
- **RoPE**: 旋转位置编码
- **Self-Attention**: 自注意力机制
- **SwiGLU**: SwiGLU 激活函数

## 编译配置

### 1. 启用 CUDA 支持
```bash
xmake f --nv-gpu=y -cv
xmake
xmake install
```

### 2. 测试 CUDA 实现
```bash
# 测试运行时
python test/test_runtime.py --device nvidia

# 测试算子
python test/test_ops.py --device nvidia

# 测试模型
python test/test_infer.py --model [model_path] --device nvidia
```

## 注意事项

1. **需要 CUDA Toolkit**: 确保安装了 CUDA Toolkit
2. **GPU 驱动**: 确保安装了 NVIDIA GPU 驱动
3. **编译器**: 需要支持 CUDA 的 C++ 编译器
4. **内存**: 确保有足够的显存

## 下一步

1. 安装 CUDA Toolkit
2. 配置编译环境
3. 实现完整的 CUDA 算子
4. 集成到模型
5. 测试和优化

## 参考资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Library](https://docs.nvidia.com/cuda/cublas/)
- [cuDNN Library](https://docs.nvidia.com/deeplearning/cudnn/)
