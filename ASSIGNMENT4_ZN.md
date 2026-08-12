# 作业 #4：NVIDIA、天数智芯与沐曦集成说明

## 后端结构

三个平台均使用相互独立的后端实现：

```text
src/device/nvidia/              NVIDIA Runtime、类型工具和资源
src/device/iluvatar/            CoreX Runtime、类型工具和资源
src/device/metax/               MXMACA Runtime 与类型工具
src/ops/<op>/nvidia/            NVIDIA 算子
src/ops/<op>/iluvatar/          天数智芯算子
src/ops/<op>/metax/             沐曦算子
xmake/nvidia.lua                NVIDIA 构建规则
xmake/iluvatar.lua              CoreX 构建规则
xmake/metax.lua                 MXMACA 构建规则
```

天数和沐曦后端都不包含或编译其他 GPU 后端目录中的文件。统一算子入口只负责按照 `DeviceType` 分发到对应命名空间。沐曦实现直接使用 `mxcc`、`mc_runtime.h` 和 `mcruntime`，不依赖 CUDA Runtime 或其他平台兼容库。

## 平台状态

| 平台 | 构建开关 | 设备类型 | 状态 |
| --- | --- | --- | --- |
| NVIDIA CUDA | `--nv-gpu=y` | `DeviceType.NVIDIA` | Runtime、算子与 Qwen2 推理链路已实现 |
| 天数智芯 CoreX | `--iluvatar-gpu=y` | `DeviceType.ILUVATAR` | Runtime、算子与 Qwen2 推理链路已实现，需在 CoreX 机器验证 |
| 沐曦 MXMACA | `--metax-gpu=y` | `DeviceType.METAX` | Runtime、算子与 Qwen2 推理链路已实现，需在沐曦机器验证 |

为防止不同 GPU SDK 的兼容 ABI 或编译工具链互相污染，同一个构建只能打开一个 GPU 后端开关。源码和构建目标保持分离，分别构建时只链接对应平台 SDK。

## CoreX 构建

```bash
export COREX_ROOT=/usr/local/corex

xmake f --iluvatar-gpu=y \
    --corex-root="$COREX_ROOT" \
    --corex-arch=ivcore10 \
    -cv
xmake
xmake install
python -m pip install --force-reinstall --no-deps ./python
```

`--corex-arch` 可根据设备改为 `ivcore11` 等 CoreX 架构。构建脚本使用 `$COREX_ROOT/bin/clang++`，并从 SDK 自身的 `include`、`lib` 或 `lib64` 查找头文件和动态库。

## MXMACA 构建

```bash
export MACA_PATH=/opt/maca

xmake f --metax-gpu=y \
    --maca-root="$MACA_PATH" \
    --maca-arch=native \
    -cv
xmake
xmake install
python -m pip install --force-reinstall --no-deps ./python
```

构建脚本使用 `$MACA_PATH/mxgpu_llvm/bin/mxcc` 编译独立的 `.maca` 源文件，并由 `mxcc` 驱动最终共享库链接，以注入 MXMACA kernel 注册与启动支持。Xmake 内部的链接适配只过滤其自动添加、但 `mxcc` 不接受的宿主 `-m64` 和 `-s` 参数。`--maca-arch` 会原样传给 `mxcc -offload-arch`，可按目标机器替换 `native`。

`xmake install` 只把共享库复制到源码树的 `python/llaisys/libllaisys/`。如果 Python 包此前以非 editable 方式安装到了 `site-packages`，必须重新执行上面的 `pip install --force-reinstall`，否则测试仍会加载旧的共享库。也可以临时设置 `PYTHONPATH="$PWD/python"` 直接使用源码树中的包。

## 实现范围

- Runtime：设备枚举、切换与同步，Stream 生命周期，Device/Host 内存，同步与异步拷贝。
- 算子：Add、Argmax、Embedding、Linear、RMSNorm、RoPE、Self-Attention、SwiGLU。
- Python：新增 `DeviceType.ILUVATAR`、`DeviceType.METAX` 以及对应命令行设备名。
- 模型：Qwen2 通过通用设备、Runtime 和算子接口接入天数及沐曦后端。

## 可选验证命令

本次按要求不运行测试。具备天数设备时可以执行：

```bash
python test/test_runtime.py --device iluvatar

for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    python "test/ops/${op}.py" --device iluvatar
done

python test/test_infer.py \
    --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    --test \
    --device iluvatar

python test/test_runtime.py --device metax

for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    python "test/ops/${op}.py" --device metax
done

python test/test_infer.py \
    --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
    --test \
    --device metax
```
