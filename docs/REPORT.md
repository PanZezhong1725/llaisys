# LLAISYS 多平台适配报告（Assignment #4）

## 一、支持的平台及其状态

| 平台 | 厂商 | 设备/环境 | 状态 |
|------|------|-----------|------|
| NVIDIA | NVIDIA | CUDA SDK（CUDA 13.3） | ✅ 已支持，runtime / 算子 / 推理均通过 |
| 天数（SUDA） | Iluvatar（天数智芯） | CoreX 4.4.0，GPU `Iluvatar BI-V150`（32GB） | ✅ 本次新增，runtime / 算子 / 推理均通过 |

> 平台选择方式：通过 `--device {cpu,nvidia,suda}` 显式指定设备类型。
>
> 关于天数平台：CoreX SDK 对外提供 **CUDA 源码级兼容层**（`cuda_runtime.h` / `cublas_v2.h` / `nvcc` / `libcudart.so` / `libcublas.so`），**没有独立的 `suda_runtime.h` / `ixblas.h` API**。因此天数后端实现为「复用 CUDA API + 独立 `suda` 命名空间/目录/device type」。

## 二、复现流程

### 通用前置
- 编译工具：Xmake、g++/gcc、Python ≥ 3.9、torch、transformers、safetensors、huggingface_hub
- 模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

### 2.1 NVIDIA 平台
```bash
xmake f --nv-gpu=y -c
xmake -y
xmake install
python3 test/test_runtime.py --device nvidia
python3 test/ops/add.py --device nvidia        # 及其余 7 个算子
python3 test/test_infer.py --model <model_dir> --test --device nvidia
```

### 2.2 天数（SUDA）平台
```bash
# 1) 配置环境变量（平台已安装 xmake v2.8.7 / python3 / torch 2.7.1）
export PATH=/usr/local/corex-4.4.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex-4.4.0/lib64:/usr/local/corex-4.4.0/lib:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/corex-4.4.0
export XMAKE_ROOT=y

# 2) 修补 CoreX 的 nvcc wrapper（解析 xmake 的 -Xcompiler/-Werror 等参数）
#    用临时 wrapbin/nvcc 覆盖 /usr/local/corex-4.4.0/bin/nvcc

# 3) 补 libcudadevrt 软链（CoreX 不提供该库，本项目不使用动态并行）
ln -sf libcudart.so /usr/local/corex-4.4.0/lib64/libcudadevrt.so

# 4) 编译 + 安装
xmake f --suda-gpu=y -c
xmake -y
xmake install          # 复制 libllaisys.so 到 python/llaisys/libllaisys/

# 5) 测试
export PYTHONPATH=<repo>/python:$PYTHONPATH
python3 test/test_runtime.py --device suda
python3 test/ops/<op>.py --device suda       # add/argmax/embedding/linear/rms_norm/rope/self_attention/swiglu
python3 test/test_qwen2_load.py <model_dir> suda
```

## 三、复现结果

### 3.1 天数（SUDA）平台实测结果（BI-V150）

| 测试项 | 结果 |
|--------|------|
| runtime：设备枚举 + memcpy | ✅ 识别 1 个设备，Passed |
| add | ✅ 全部 dtype（f32/f16/bf16）通过 |
| argmax | ✅ 通过 |
| embedding | ✅ 通过 |
| linear | ✅ 通过 |
| rms_norm | ✅ 通过 |
| rope | ✅ 通过（角度计算改用 float32）|
| self_attention | ✅ 通过 |
| swiglu | ✅ 通过 |
| 端到端推理（DeepSeek-R1-Distill-Qwen-1.5B） | ✅ 生成文本正确，约 37.46 tokens/sec |

端到端推理输出示例（prompt “Hello, who are you?”）：
```
Generated text: <｜User｜>Hello, who are you?<｜Assistant｜><think>
I'm DeepSeek-R1, an AI assistant created exclusively by the Chinese Company DeepSeek. I specialize in helping you tackle complex STEM challenges through analytical thinking,
```

### 3.2 NVIDIA 平台状态
NVIDIA 后端为本次新增天数后端所复用的基线，此前已实现并验证通过（runtime / 8 个算子 / 端到端推理）。

## 四、天数平台适配要点（与标准 CUDA 的差异）

1. **`cublasGemmEx` 签名差异**：CoreX 移植版第 18 个参数 `computeType` 类型为 `cudaDataType`（标准 CUDA 为 `cublasComputeType_t`），须传 `CUDA_R_32F` 而非 `CUBLAS_COMPUTE_32F`。
2. **无 float64（double）支持**：ivcore 设备端 double 数学不可用（torch 明确警告 `Limited support for torch.double`），rope 的角度计算改用 `float`。
3. **缺 `libcudadevrt.so`**：xmake CUDA 规则默认链接它，CoreX 不提供，用软链指向 `libcudart.so` 规避。

## 五、限制说明

- `test/test_infer.py --test` 的逐 token 对齐模式需同时加载 HF 参考模型，HF 模型在 CoreX 上因缺失 float64 GPU 支持而无法工作（平台限制，非本项目代码问题）。天数平台的端到端验证使用 `test_qwen2_load.py --device suda`（纯 LLAISYS 后端）完成。
- 两个平台通过独立的 `xmake f --nv-gpu=y` / `--suda-gpu=y` 开关互不干扰，可分别编译。