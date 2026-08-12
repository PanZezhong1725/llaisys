# 作业 #4 完成报告：NVIDIA、天数智芯与沐曦平台集成

## 1. 完成结论

本作业要求在框架中集成至少两类 CUDA-like 加速平台。本项目已完成 NVIDIA CUDA、天数智芯 CoreX 和沐曦 MXMACA 三套后端的接入，覆盖设备运行时、基础算子、Python 设备选择和 Qwen2 推理调用链。

三套后端彼此独立：平台相关的 Runtime、资源管理和 Kernel 分别位于各自目录中，不依赖其他 GPU 后端的实现。编译时一次只选择一个 GPU 后端，最终都通过统一的 `llaisys` C API 和 Python API 对外提供能力。

| 平台 | 构建选项 | Python 设备名 | Runtime | 8 个算子 | Qwen2 调用链 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- |
| NVIDIA CUDA | `--nv-gpu=y` | `nvidia` | 已完成 | 已完成 | 已接入 | 实现完成，本文给出 NV 环境复现步骤 |
| 天数智芯 CoreX | `--iluvatar-gpu=y` | `iluvatar` | 已完成 | 已完成 | 已接入 | 实现完成，未在天数设备上实机验收 |
| 沐曦 MXMACA | `--metax-gpu=y` | `metax` | 已完成 | 已完成 | 已接入 | 实现完成，需在沐曦设备上实机验收 |

> 按本次任务要求，本报告不代跑测试。下文提供从清理、编译、安装到 Runtime、算子和模型验证的完整复现命令；表中的“已完成”表示代码实现和调用链已接入，不等同于声明三类硬件均已完成本次实机测试。

## 2. 后端隔离与统一接口

平台代码按目录分离：

```text
src/device/nvidia/       # NVIDIA CUDA Runtime 与资源管理
src/device/iluvatar/     # 天数智芯 CoreX Runtime 与资源管理
src/device/metax/        # 沐曦 MXMACA Runtime 与资源管理

src/ops/*/nvidia/        # NVIDIA 算子实现
src/ops/*/iluvatar/      # 天数智芯算子实现
src/ops/*/metax/         # 沐曦算子实现
```

公共层只负责设备枚举和分发：

- C/C++ 设备枚举：`LLAISYS_DEVICE_NVIDIA`、`LLAISYS_DEVICE_ILUVATAR`、`LLAISYS_DEVICE_METAX`。
- Python 设备枚举：`DeviceType.NVIDIA`、`DeviceType.ILUVATAR`、`DeviceType.METAX`。
- Runtime 和算子入口根据设备类型转发到对应命名空间。
- Qwen2 模型层使用统一 Tensor、Runtime 和算子接口，不直接依赖某个平台 SDK。
- `xmake.lua` 约束一次构建只启用一个 GPU 后端，避免多个平台工具链和运行库相互污染。

因此，天数智芯和沐曦后端不会复用或依赖 NVIDIA Kernel；三者只共享框架的公共接口定义。

## 3. 已完成的实现

### 3.1 NVIDIA CUDA 后端

NVIDIA 后端实现位于 `src/device/nvidia/`、`src/ops/*/nvidia/` 和 `xmake/nvidia.lua`，完成内容如下：

- Runtime：设备数量查询、设备切换、设备同步、流创建/销毁/同步。
- 内存：设备内存分配与释放、Pinned Host 内存分配与释放。
- 数据传输：同步和异步 `cudaMemcpy`，支持 Host/Device 方向转换。
- 资源管理：按线程和设备缓存 cuBLAS Handle，并将 Handle 绑定到当前 CUDA Stream。
- 算子：`add`、`argmax`、`embedding`、`linear`、`rms_norm`、`rope`、`self_attention`、`swiglu`。
- 数据类型：算子路径覆盖 `F32`、`F16` 和 `BF16`；`linear` 使用 `cublasGemmEx` 并采用 FP32 累加。
- 构建：独立生成 NVIDIA device/ops 静态库，启用 CUDA device link，链接 `cublas`，默认针对本机 GPU 架构生成代码。
- 上层接入：Python 测试入口支持 `--device nvidia`，Qwen2 可通过统一接口运行在 NVIDIA 设备上。

### 3.2 天数智芯 CoreX 后端

天数智芯实现位于 `src/device/iluvatar/`、`src/ops/*/iluvatar/` 和 `xmake/iluvatar.lua`：

- Runtime、流、内存及同步/异步拷贝已接入 CoreX Runtime。
- 8 个算子均有独立的天数智芯实现。
- GEMM 使用 `cuinfer`，并通过平台资源管理层维护 Handle 与 Stream。
- Python 和 Qwen2 调用链接受 `iluvatar` 设备选择。

### 3.3 沐曦 MXMACA 后端

沐曦实现位于 `src/device/metax/`、`src/ops/*/metax/` 和 `xmake/metax.lua`：

- Runtime、流、内存及同步/异步拷贝已接入 MXMACA Runtime。
- 8 个算子均有独立的沐曦实现。
- GEMM 使用 `mcblas`，并通过平台资源管理层维护 Handle 与 Stream。
- 使用 MXCC 编译设备代码；共享库链接阶段由专用脚本过滤 MXCC 不支持的宿主链接参数。
- Python 和 Qwen2 调用链接受 `metax` 设备选择。

## 4. NVIDIA 环境完整复现步骤

以下步骤适用于当前 NVIDIA 环境，命令均从项目根目录执行。

### 4.1 检查基础环境

```bash
cd /data/llaisys
nvidia-smi
nvcc --version
xmake --version
python --version
```

应能识别 NVIDIA 驱动、CUDA Toolkit、Xmake 和 Python。若项目实际位于其他目录，只需将 `/data/llaisys` 替换为实际路径。

### 4.2 清理并配置 NVIDIA 构建

```bash
xmake clean
xmake f -c --nv-gpu=y -cv
```

`-c` 会清除旧的 Xmake 配置缓存，避免此前选择的天数智芯或沐曦工具链残留。配置结果中应启用 NVIDIA 后端，且不应同时启用另外两个 GPU 后端。

### 4.3 编译并安装 C++ 动态库

```bash
xmake
xmake install
```

编译完成后，核心产物为：

```text
build/linux/x86_64/release/libllaisys.so
```

### 4.4 重新安装 Python 包

```bash
python -m pip install --force-reinstall --no-cache-dir --no-deps ./python
```

这一步必须在 `xmake install` 之后执行，否则 Python 可能继续加载 `site-packages` 中旧版本的 `libllaisys.so`，从而出现“代码已经修改，但运行时仍报告旧符号缺失”的问题。

可用以下命令确认实际加载位置：

```bash
python -c "import llaisys; print(llaisys.__file__)"
python -c "from llaisys.libllaisys import LIB_LLAISYS; print(LIB_LLAISYS._name)"
```

### 4.5 Runtime 复现

```bash
python test/test_runtime.py --device nvidia
```

该命令覆盖设备发现、设备切换、流管理、内存分配、同步拷贝和异步拷贝等 Runtime 路径。

### 4.6 算子复现

```bash
for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    python "test/ops/${op}.py" --device nvidia
done
```

该循环依次验证本作业要求的 8 个算子。若需要定位单个算子，可直接运行对应脚本，例如：

```bash
python test/ops/linear.py --device nvidia
```

### 4.7 Qwen2 推理复现

```bash
python test/test_infer.py \
    --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    --test \
    --device nvidia
```

将模型路径替换为本机实际路径。该命令用于验证从 Python、模型层、Tensor/Runtime 到 NVIDIA 算子的完整调用链。

## 5. 天数智芯 CoreX 复现步骤

```bash
export COREX_ROOT=/usr/local/corex

xmake clean
xmake f -c --iluvatar-gpu=y -cv
xmake
xmake install
python -m pip install --force-reinstall --no-cache-dir --no-deps ./python

python test/test_runtime.py --device iluvatar
for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    python "test/ops/${op}.py" --device iluvatar
done
python test/test_infer.py \
    --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    --test \
    --device iluvatar
```

如 CoreX 安装路径不同，应将 `COREX_ROOT` 修改为目标机器的实际 SDK 路径。

## 6. 沐曦 MXMACA 复现步骤

```bash
export MACA_PATH=/opt/maca

xmake clean
xmake f -c --metax-gpu=y -cv
xmake
xmake install
python -m pip install --force-reinstall --no-cache-dir --no-deps ./python

python test/test_runtime.py --device metax
for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    python "test/ops/${op}.py" --device metax
done
python test/test_infer.py \
    --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    --test \
    --device metax
```

如 MXMACA 安装路径不同，应将 `MACA_PATH` 修改为目标机器的实际 SDK 路径。

## 7. 验收与故障排查

### 7.1 推荐验收顺序

建议按以下顺序验收，每一步通过后再进入下一步：

1. `xmake f` 能识别所选平台工具链和 SDK。
2. `xmake` 能生成 `libllaisys.so`，加载时不存在未定义符号。
3. Runtime 测试通过，确认设备、流、内存和拷贝链路可用。
4. 8 个算子测试分别通过。
5. Qwen2 测试模式通过，确认模型端到端调用链可用。

### 7.2 动态库未定义符号

若导入 Python 包时出现类似下面的错误：

```text
OSError: .../libllaisys.so: undefined symbol: ...
```

先确认已使用对应平台重新配置并完整编译，再执行：

```bash
xmake clean
xmake f -c --nv-gpu=y -cv    # 按目标平台替换此选项
xmake
xmake install
python -m pip install --force-reinstall --no-cache-dir --no-deps ./python
```

随后通过 `llaisys.__file__` 和 `LIB_LLAISYS._name` 检查 Python 包及动态库是否仍在加载旧安装。

### 7.3 多后端切换

从一个 GPU 平台切换到另一个平台时必须重新执行 `xmake clean` 和 `xmake f -c`。不要在同一次构建中同时设置 `--nv-gpu=y`、`--iluvatar-gpu=y` 或 `--metax-gpu=y`。

## 8. 运行日志

### NVIDIA CUDA

```text
`torch_dtype` is deprecated! Use `dtype` instead!
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[INFO] llaisys.models.qwen2: Loading weight shard: model.safetensors
[INFO] llaisys.models.qwen2: Loaded 339 tensors from 1 shards
[Qwen2] forward complete: q_len=10 past_len=0 total_len=10 next_token=91786
[Qwen2] forward complete: q_len=1 past_len=10 total_len=11 next_token=0
[Qwen2] forward complete: q_len=1 past_len=11 total_len=12 next_token=358
[Qwen2] forward complete: q_len=1 past_len=12 total_len=13 next_token=2776
[Qwen2] forward complete: q_len=1 past_len=13 total_len=14 next_token=18183
[Qwen2] forward complete: q_len=1 past_len=14 total_len=15 next_token=39350
[Qwen2] forward complete: q_len=1 past_len=15 total_len=16 next_token=10911
[Qwen2] forward complete: q_len=1 past_len=16 total_len=17 next_token=16
[Qwen2] forward complete: q_len=1 past_len=17 total_len=18 next_token=11
[Qwen2] forward complete: q_len=1 past_len=18 total_len=19 next_token=458
[Qwen2] forward complete: q_len=1 past_len=19 total_len=20 next_token=20443
[Qwen2] forward complete: q_len=1 past_len=20 total_len=21 next_token=11229
[Qwen2] forward complete: q_len=1 past_len=21 total_len=22 next_token=17847
[Qwen2] forward complete: q_len=1 past_len=22 total_len=23 next_token=3465
[Qwen2] forward complete: q_len=1 past_len=23 total_len=24 next_token=553
[Qwen2] forward complete: q_len=1 past_len=24 total_len=25 next_token=18183
[Qwen2] forward complete: q_len=1 past_len=25 total_len=26 next_token=39350
[Qwen2] forward complete: q_len=1 past_len=26 total_len=27 next_token=13
[Qwen2] forward complete: q_len=1 past_len=27 total_len=28 next_token=358
[Qwen2] forward complete: q_len=1 past_len=28 total_len=29 next_token=2776
[Qwen2] forward complete: q_len=1 past_len=29 total_len=30 next_token=518
[Qwen2] forward complete: q_len=1 past_len=30 total_len=31 next_token=697
[Qwen2] forward complete: q_len=1 past_len=31 total_len=32 next_token=2473
[Qwen2] forward complete: q_len=1 past_len=32 total_len=33 next_token=323
[Qwen2] forward complete: q_len=1 past_len=33 total_len=34 next_token=1035
[Qwen2] forward complete: q_len=1 past_len=34 total_len=35 next_token=387
[Qwen2] forward complete: q_len=1 past_len=35 total_len=36 next_token=33972
[Qwen2] forward complete: q_len=1 past_len=36 total_len=37 next_token=311
[Qwen2] forward complete: q_len=1 past_len=37 total_len=38 next_token=7789
[Qwen2] forward complete: q_len=1 past_len=38 total_len=39 next_token=498
[Qwen2] forward complete: q_len=1 past_len=39 total_len=40 next_token=448
[Qwen2] forward complete: q_len=1 past_len=40 total_len=41 next_token=894
[Qwen2] forward complete: q_len=1 past_len=41 total_len=42 next_token=43883
[Qwen2] forward complete: q_len=1 past_len=42 total_len=43 next_token=476
[Qwen2] forward complete: q_len=1 past_len=43 total_len=44 next_token=9079
[Qwen2] forward complete: q_len=1 past_len=44 total_len=45 next_token=498
[Qwen2] forward complete: q_len=1 past_len=45 total_len=46 next_token=1231
[Qwen2] forward complete: q_len=1 past_len=46 total_len=47 next_token=614
[Qwen2] forward complete: q_len=1 past_len=47 total_len=48 next_token=624
[Qwen2] forward complete: q_len=1 past_len=48 total_len=49 next_token=151649
[Qwen2] forward complete: q_len=1 past_len=49 total_len=50 next_token=271
[Qwen2] forward complete: q_len=1 past_len=50 total_len=51 next_token=91786
[Qwen2] forward complete: q_len=1 past_len=51 total_len=52 next_token=0
[Qwen2] forward complete: q_len=1 past_len=52 total_len=53 next_token=358
[Qwen2] forward complete: q_len=1 past_len=53 total_len=54 next_token=2776
[Qwen2] forward complete: q_len=1 past_len=54 total_len=55 next_token=18183
[Qwen2] forward complete: q_len=1 past_len=55 total_len=56 next_token=39350
[Qwen2] forward complete: q_len=1 past_len=56 total_len=57 next_token=10911
[Qwen2] forward complete: q_len=1 past_len=57 total_len=58 next_token=16
[Qwen2] forward complete: q_len=1 past_len=58 total_len=59 next_token=11
[Qwen2] forward complete: q_len=1 past_len=59 total_len=60 next_token=458
[Qwen2] forward complete: q_len=1 past_len=60 total_len=61 next_token=20443
[Qwen2] forward complete: q_len=1 past_len=61 total_len=62 next_token=11229
[Qwen2] forward complete: q_len=1 past_len=62 total_len=63 next_token=17847
[Qwen2] forward complete: q_len=1 past_len=63 total_len=64 next_token=3465
[Qwen2] forward complete: q_len=1 past_len=64 total_len=65 next_token=553
[Qwen2] forward complete: q_len=1 past_len=65 total_len=66 next_token=18183
[Qwen2] forward complete: q_len=1 past_len=66 total_len=67 next_token=39350
[Qwen2] forward complete: q_len=1 past_len=67 total_len=68 next_token=13
[Qwen2] forward complete: q_len=1 past_len=68 total_len=69 next_token=358
[Qwen2] forward complete: q_len=1 past_len=69 total_len=70 next_token=2776
[Qwen2] forward complete: q_len=1 past_len=70 total_len=71 next_token=518
[Qwen2] forward complete: q_len=1 past_len=71 total_len=72 next_token=697
[Qwen2] forward complete: q_len=1 past_len=72 total_len=73 next_token=2473
[Qwen2] forward complete: q_len=1 past_len=73 total_len=74 next_token=323
[Qwen2] forward complete: q_len=1 past_len=74 total_len=75 next_token=1035
[Qwen2] forward complete: q_len=1 past_len=75 total_len=76 next_token=387
[Qwen2] forward complete: q_len=1 past_len=76 total_len=77 next_token=33972
[Qwen2] forward complete: q_len=1 past_len=77 total_len=78 next_token=311
[Qwen2] forward complete: q_len=1 past_len=78 total_len=79 next_token=7789
[Qwen2] forward complete: q_len=1 past_len=79 total_len=80 next_token=498
[Qwen2] forward complete: q_len=1 past_len=80 total_len=81 next_token=448
[Qwen2] forward complete: q_len=1 past_len=81 total_len=82 next_token=894
[Qwen2] forward complete: q_len=1 past_len=82 total_len=83 next_token=43883
[Qwen2] forward complete: q_len=1 past_len=83 total_len=84 next_token=476
[Qwen2] forward complete: q_len=1 past_len=84 total_len=85 next_token=9079
[Qwen2] forward complete: q_len=1 past_len=85 total_len=86 next_token=498
[Qwen2] forward complete: q_len=1 past_len=86 total_len=87 next_token=1231
[Qwen2] forward complete: q_len=1 past_len=87 total_len=88 next_token=614
[Qwen2] forward complete: q_len=1 past_len=88 total_len=89 next_token=13
[Qwen2] forward complete: q_len=1 past_len=89 total_len=90 next_token=151643
Loading model from local path: /home/wnyxvo/huggingface/DeepSeek-R1-Distill-Qwen-1.5B

=== Answer ===

Tokens:
[151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 624, 151649, 271, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 13, 151643]

Contents:
<｜User｜>Who are you?<｜Assistant｜><think>
Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.
</think>

Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.


Time elapsed: 4.65s


=== Your Result ===

Tokens:
[151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 624, 151649, 271, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 13, 151643]

Contents:
<｜User｜>Who are you?<｜Assistant｜><think>
Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.
</think>

Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.


Time elapsed: 2.97s

[92mTest passed![0m

Qwen2 KV cache resized: capacity=256, valid_length=0

```

### MXMACA

```text

`torch_dtype` is deprecated! Use `dtype` instead!
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
/opt/conda/lib/python3.10/site-packages/torch/nn/functional.py:6001: UserWarning: 1Torch was not compiled with memory efficient attention. (Triggered internally at /workspace/framework/mcPytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:738.)
  return _scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale = scale, enable_gqa = enable_gqa)
[INFO] llaisys.models.qwen2: Loading weight shard: model.safetensors
[INFO] llaisys.models.qwen2: Loaded 339 tensors from 1 shards
[Qwen2] forward complete: q_len=10 past_len=0 total_len=10 next_token=91786
[Qwen2] forward complete: q_len=1 past_len=10 total_len=11 next_token=0
[Qwen2] forward complete: q_len=1 past_len=11 total_len=12 next_token=358
[Qwen2] forward complete: q_len=1 past_len=12 total_len=13 next_token=2776
[Qwen2] forward complete: q_len=1 past_len=13 total_len=14 next_token=18183
[Qwen2] forward complete: q_len=1 past_len=14 total_len=15 next_token=39350
[Qwen2] forward complete: q_len=1 past_len=15 total_len=16 next_token=10911
[Qwen2] forward complete: q_len=1 past_len=16 total_len=17 next_token=16
[Qwen2] forward complete: q_len=1 past_len=17 total_len=18 next_token=11
[Qwen2] forward complete: q_len=1 past_len=18 total_len=19 next_token=458
[Qwen2] forward complete: q_len=1 past_len=19 total_len=20 next_token=20443
[Qwen2] forward complete: q_len=1 past_len=20 total_len=21 next_token=11229
[Qwen2] forward complete: q_len=1 past_len=21 total_len=22 next_token=17847
[Qwen2] forward complete: q_len=1 past_len=22 total_len=23 next_token=3465
[Qwen2] forward complete: q_len=1 past_len=23 total_len=24 next_token=553
[Qwen2] forward complete: q_len=1 past_len=24 total_len=25 next_token=18183
[Qwen2] forward complete: q_len=1 past_len=25 total_len=26 next_token=39350
[Qwen2] forward complete: q_len=1 past_len=26 total_len=27 next_token=13
[Qwen2] forward complete: q_len=1 past_len=27 total_len=28 next_token=358
[Qwen2] forward complete: q_len=1 past_len=28 total_len=29 next_token=2776
[Qwen2] forward complete: q_len=1 past_len=29 total_len=30 next_token=518
[Qwen2] forward complete: q_len=1 past_len=30 total_len=31 next_token=697
[Qwen2] forward complete: q_len=1 past_len=31 total_len=32 next_token=2473
[Qwen2] forward complete: q_len=1 past_len=32 total_len=33 next_token=323
[Qwen2] forward complete: q_len=1 past_len=33 total_len=34 next_token=1035
[Qwen2] forward complete: q_len=1 past_len=34 total_len=35 next_token=387
[Qwen2] forward complete: q_len=1 past_len=35 total_len=36 next_token=33972
[Qwen2] forward complete: q_len=1 past_len=36 total_len=37 next_token=311
[Qwen2] forward complete: q_len=1 past_len=37 total_len=38 next_token=7789
[Qwen2] forward complete: q_len=1 past_len=38 total_len=39 next_token=498
[Qwen2] forward complete: q_len=1 past_len=39 total_len=40 next_token=448
[Qwen2] forward complete: q_len=1 past_len=40 total_len=41 next_token=894
[Qwen2] forward complete: q_len=1 past_len=41 total_len=42 next_token=43883
[Qwen2] forward complete: q_len=1 past_len=42 total_len=43 next_token=476
[Qwen2] forward complete: q_len=1 past_len=43 total_len=44 next_token=9079
[Qwen2] forward complete: q_len=1 past_len=44 total_len=45 next_token=498
[Qwen2] forward complete: q_len=1 past_len=45 total_len=46 next_token=1231
[Qwen2] forward complete: q_len=1 past_len=46 total_len=47 next_token=614
[Qwen2] forward complete: q_len=1 past_len=47 total_len=48 next_token=624
[Qwen2] forward complete: q_len=1 past_len=48 total_len=49 next_token=151649
[Qwen2] forward complete: q_len=1 past_len=49 total_len=50 next_token=271
[Qwen2] forward complete: q_len=1 past_len=50 total_len=51 next_token=91786
[Qwen2] forward complete: q_len=1 past_len=51 total_len=52 next_token=0
[Qwen2] forward complete: q_len=1 past_len=52 total_len=53 next_token=358
[Qwen2] forward complete: q_len=1 past_len=53 total_len=54 next_token=2776
[Qwen2] forward complete: q_len=1 past_len=54 total_len=55 next_token=18183
[Qwen2] forward complete: q_len=1 past_len=55 total_len=56 next_token=39350
[Qwen2] forward complete: q_len=1 past_len=56 total_len=57 next_token=10911
[Qwen2] forward complete: q_len=1 past_len=57 total_len=58 next_token=16
[Qwen2] forward complete: q_len=1 past_len=58 total_len=59 next_token=11
[Qwen2] forward complete: q_len=1 past_len=59 total_len=60 next_token=458
[Qwen2] forward complete: q_len=1 past_len=60 total_len=61 next_token=20443
[Qwen2] forward complete: q_len=1 past_len=61 total_len=62 next_token=11229
[Qwen2] forward complete: q_len=1 past_len=62 total_len=63 next_token=17847
[Qwen2] forward complete: q_len=1 past_len=63 total_len=64 next_token=3465
[Qwen2] forward complete: q_len=1 past_len=64 total_len=65 next_token=553
[Qwen2] forward complete: q_len=1 past_len=65 total_len=66 next_token=18183
[Qwen2] forward complete: q_len=1 past_len=66 total_len=67 next_token=39350
[Qwen2] forward complete: q_len=1 past_len=67 total_len=68 next_token=13
[Qwen2] forward complete: q_len=1 past_len=68 total_len=69 next_token=358
[Qwen2] forward complete: q_len=1 past_len=69 total_len=70 next_token=2776
[Qwen2] forward complete: q_len=1 past_len=70 total_len=71 next_token=518
[Qwen2] forward complete: q_len=1 past_len=71 total_len=72 next_token=697
[Qwen2] forward complete: q_len=1 past_len=72 total_len=73 next_token=2473
[Qwen2] forward complete: q_len=1 past_len=73 total_len=74 next_token=323
[Qwen2] forward complete: q_len=1 past_len=74 total_len=75 next_token=1035
[Qwen2] forward complete: q_len=1 past_len=75 total_len=76 next_token=387
[Qwen2] forward complete: q_len=1 past_len=76 total_len=77 next_token=33972
[Qwen2] forward complete: q_len=1 past_len=77 total_len=78 next_token=311
[Qwen2] forward complete: q_len=1 past_len=78 total_len=79 next_token=7789
[Qwen2] forward complete: q_len=1 past_len=79 total_len=80 next_token=498
[Qwen2] forward complete: q_len=1 past_len=80 total_len=81 next_token=448
[Qwen2] forward complete: q_len=1 past_len=81 total_len=82 next_token=894
[Qwen2] forward complete: q_len=1 past_len=82 total_len=83 next_token=43883
[Qwen2] forward complete: q_len=1 past_len=83 total_len=84 next_token=476
[Qwen2] forward complete: q_len=1 past_len=84 total_len=85 next_token=9079
[Qwen2] forward complete: q_len=1 past_len=85 total_len=86 next_token=498
[Qwen2] forward complete: q_len=1 past_len=86 total_len=87 next_token=1231
[Qwen2] forward complete: q_len=1 past_len=87 total_len=88 next_token=614
[Qwen2] forward complete: q_len=1 past_len=88 total_len=89 next_token=13
[Qwen2] forward complete: q_len=1 past_len=89 total_len=90 next_token=151643
Loading model from local path: /data/huggingface/DeepSeek-R1-Distill-Qwen-1.5B

=== Answer ===

Tokens:
[151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 624, 151649, 271, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 13, 151643]

Contents:
<｜User｜>Who are you?<｜Assistant｜><think>
Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.
</think>

Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.


Time elapsed: 4.85s


=== Your Result ===

Tokens:
[151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 624, 151649, 271, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894, 43883, 476, 9079, 498, 1231, 614, 13, 151643]

Contents:
<｜User｜>Who are you?<｜Assistant｜><think>
Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.
</think>

Greetings! I'm DeepSeek-R1, an artificial intelligence assistant created by DeepSeek. I'm at your service and would be delighted to assist you with any inquiries or tasks you may have.


Time elapsed: 30.38s

[92mTest passed![0m

Qwen2 KV cache resized: capacity=256, valid_length=0

```

## 9. 提交内容总结

- 完成 NVIDIA、天数智芯和沐曦三套独立 GPU 后端。
- 每套后端均接入 Runtime、内存/流管理和 8 个指定算子。
- 完成 Python 设备枚举、测试脚本设备参数和 Qwen2 模型调用链接入。
- 完成三个平台各自的 Xmake 构建配置与依赖链接。
- 提供 NVIDIA 当前环境的完整复现步骤，并保留天数智芯、沐曦目标机器的复现命令。
- 本次未执行测试；最终实机结果应以对应平台按上述步骤运行得到的输出为准。
