# LLAISYS 作业 #4 报告

## 总结

本次 PR 完成作业 #4：为 LLAISYS 集成 CUDA。

实现内容包括：

- CUDA Runtime API：设备管理、stream、显存分配、内存拷贝
- NVIDIA 设备资源管理，包含 cuBLAS handle
- 全部作业算子的 CUDA 实现：`add`、`argmax`、`embedding`、`linear`、`rearrange`、`rms_norm`、`rope`、`self_attention`、`swiglu`
- Qwen2 模型通过统一 Tensor/算子分发支持 CUDA 推理
- Xmake 构建配置支持 NVIDIA（`--nv-gpu=y`）和 MetaX MACA（`--metax-gpu=y`）

## 支持平台与状态

| 平台 | 状态 |
| --- | --- |
| CPU | 通过。CI 中 Assignment #0-#3 全部通过。 |
| NVIDIA RTX 4090 | 通过。Runtime、全部算子测试、完整模型推理测试均通过。 |
| MetaX C500（MACA） | 构建配置和源码已就绪，等待真实硬件验证。 |
| MetaX C500（MACA） | 通过。使用 Makefile.metax 构建，Runtime、全部算子测试和完整模型推理均通过。 |

## 复现步骤

### RTX 4090

测试环境：

- GPU：NVIDIA GeForce RTX 4090 D
- 驱动：570.124.06
- CUDA Toolkit：12.8
- Python：3.12
- PyTorch：NVIDIA 版，支持 CUDA 12.8
- Xmake：3.1.0

构建：

```bash
xmake f --nv-gpu=y -c
xmake -r
xmake install
```

测试：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/python

python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia
python test/test_infer.py --model [model_path] --test --device nvidia
```

### MetaX C500（待硬件验证）
### MetaX C500（MACA）

测试环境：

- MACA：3.5.3
- 设备架构：xcore1000
- Python：conda 3.10，torch 2.8.0+metax3.5.3.9

使用独立的 MetaX Makefile 构建：

```bash
make -f Makefile.metax -j4
make -f Makefile.metax install
```

测试命令仍使用 `--device nvidia`，因为 MetaX 通过 MACA 的 CUDA 兼容 Runtime 复用 `LLAISYS_DEVICE_NVIDIA` 设备类型：

```bash
export MACA_PATH=/opt/maca-3.5.3
export MACA_HOME=/opt/maca-3.5.3
export LD_LIBRARY_PATH=/opt/maca-3.5.3/lib64:/opt/maca-3.5.3/mxgpu_llvm/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/python

python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia
python test/test_infer.py --test --device nvidia
```

## 测试结果

### CI

最新 Actions 已通过：

- Windows latest：成功
- Ubuntu latest：成功
- Assignment #0、#1、#2、#3 全部通过

### RTX 4090

- Runtime 测试：通过
- 算子测试：8 个算子全部通过，覆盖 F32 / F16 / BF16
- 模型推理：`test_infer.py --test --device nvidia` 通过，128 个 token 与 PyTorch 输出完全一致

### MetaX C500

- Runtime 测试：通过
- 算子测试：8 个算子全部通过，覆盖 F32 / F16 / BF16
- 模型推理：`test_infer.py --test --device nvidia` 通过，128 个 token 与 PyTorch 输出完全一致

## 关键修复

- 新增 `src/llaisys/cuda_link.cu`，让共享库使用 nvcc 完成 device link；否则加载 `libllaisys.so` 会报未定义的 `__cudaRegisterLinkedBinary` 符号。
- Linear 的 F16/BF16 改为 `cublasGemmEx` + `CUBLAS_COMPUTE_32F`，使用 FP32 累加。
- 修复 CUDA self-attention 在 `kvlen > qlen` 时的 causal mask 偏移：`key_limit = q_idx + (kvlen - qlen)`。
- 修复 CUDA argmax 二级归约：当 block 数量超过 256 时，改用 grid-stride 循环处理所有 block 结果。
- 修复 `test/ops/self_attention.py`，causal mask 改为在 query 所在设备上创建。
- 新增 `Makefile.metax`，使用 mxcc 的 `--maca-link -fgpu-rdc` 并链接 `-lruntime_cu` 完成 MetaX 独立构建。
