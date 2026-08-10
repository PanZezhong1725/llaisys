# LLAISYS Assignment #4 报告

CUDA 集成 + 双平台适配（NVIDIA、天数智芯 Iluvatar CoreX），以及性能优化。

## 1. 环境

**NVIDIA（本地开发机，WSL2）**
- GPU：RTX 5070 Ti Laptop GPU（`sm_120`，Blackwell）
- CUDA：12.9

**天数智芯 Iluvatar CoreX（远程云平台）**
- 硬件：Iluvatar BI-V150，SDK `corex-4.4.0`
- 编译器：`/usr/local/corex/bin/clang++`（用 `-x ivcore` 而非标准 `-x cuda`）
- cuBLAS：`libcublas.so.10.2.3.254`（CUDA 10.2 时代兼容库）

项目不依赖 cuDNN（早期探索过 cuDNN 加速 `self_attention`/`argmax`/`rope`，后详见第 5 节，已全部移除）。

## 2. 复现步骤

**NVIDIA：**
```bash
xmake f --nv-gpu=y && xmake && xmake install
pip install -e ./python

python test/ops/<op>.py --device nvidia
python test/test_infer.py --model <dir_path/to/model> --test --device nvidia
```

**Iluvatar（远程机器）：**
```bash
export XMAKE_ROOT=y   # 容器内是 root，xmake 默认拒绝 root 运行
xmake f --iluvatar-gpu=y && xmake && xmake install
pip install -e ./python

python test/ops/<op>.py --device iluvatar
```
远程机器需先安装 xmake：`curl -fsSL https://xmake.io/shget.text | bash`（不要带 `--branch` 参数）。

## 3. 逐算子正确性

| 算子 | NVIDIA | Iluvatar |
| --- | --- | --- |
| add | ✅ | ✅ |
| embedding | ✅ | ✅ |
| argmax | ✅ | ✅ |
| rope | ✅ | ✅ |
| linear | ✅（cuBLAS） | ✅（见下方 bf16 说明） |
| swiglu | ✅ | ✅ |
| rms_norm | ✅ | ✅ |
| self_attention | ✅（V1 kernel + NVIDIA 上的 flash attention，见第 5 节） | ✅（V1 kernel） |
| rearrange | 不实现（作业 #2 遗留的废弃算子） | 同左 |

**Iluvatar `linear` 的 bf16 修复**：Iluvatar 的 cuBLAS 兼容库对 `cublasSgemmEx` 的 `CUDA_R_16BF` 返回 `CUBLAS_STATUS_NOT_SUPPORTED`。改用更通用的 `cublasGemmEx`（`computeType` 传旧式的 `CUDA_R_32F` 而非新枚举 `CUBLAS_COMPUTE_32F`）后数值完全正确，所有 shape 通过。

## 4. 完整端到端推理（`test_infer.py --test`）

| 平台 | 状态 |
| --- | --- |
| CPU | ✅ |
| NVIDIA | ✅ |
| Iluvatar | ✅（远程机器上跑通，见下方说明） |

**修复过的 bug**：`qwen2.cc` 里 `argmax` 结果 `max_idx` 在 GPU 上是设备指针，早期代码直接在主机端解引用导致段错误（只在 CPU 设备下"凑巧"能跑，因为 CPU 的显存指针和主机指针是同一地址空间）。参照 `Tensor::debug()` 的做法，改为 `memcpy_sync(..., LLAISYS_MEMCPY_D2H)` 读回主机端后再使用。修复后 32 个 token 逐个匹配 HF 参考实现。

**Iluvatar 端到端验证**：去掉 cuDNN 加速（第 5.1 节）之后，这个改动此前只在 8 个算子的单测层面验证过，完整模型推理没有在天数机器上跑过。原来的远程实例中途被销毁，重新开了一台同镜像的新实例后补跑：`test/test_infer.py --device iluvatar --test`（模型权重从 HuggingFace 自动下载）8 个算子 + 完整推理全部通过，token 输出逐个匹配 HF 参考实现，确认 cuDNN 移除对 Iluvatar 没有引入回归。

## 5. 性能优化：`self_attention`

### 5.1 cuDNN 加速路径：探索后放弃

最初尝试用 cuDNN `cudnn_frontend` SDPA Graph API 加速 decode 阶段的 `self_attention`。定位到瓶颈：SDPA 图**每次调用都从头构建**（validate → build_operation_graph → create_execution_plans → build_plans），对 decode 这种小计算量、高调用频次的场景，建图开销远大于实际算力开销——decode 从 640ms/token 降到能接受的水平，关键卡点是 K/V 的 `total_len` 每步都变，精确 shape 缓存永远不命中，而 cuDNN 的动态 shape/ragged-KV 机制在这个版本要么被 SDPA 引擎拒绝要么被注释掉未开放，分桶方案又需要额外的显式 padding mask（因果对齐会把 padding 误判为真实 token），没有把这部分设计清楚。

加上版本/平台脆弱性（本机 cuDNN 9.24.0 的 SDPA 执行引擎在 `sm_120` 上运行时崩溃，Iluvatar 的 cuDNN 7.6.5 更是没有 Graph API），决定整体放弃 cuDNN 加速，回归手写 kernel 路线——这也更贴近真实推理引擎（vLLM、TensorRT-LLM）的做法：attention kernel 本身是手写/模板化的，`total_len` 是运行时参数，不存在"建图"这一步。

去掉 cuDNN 后单纯用 V1 手写 kernel，decode 已经从 640ms/token 降到 78ms/token（约 8 倍）。

### 5.2 Flash Attention 手写 kernel（NVIDIA，prefill + decode）

在 V1 kernel 基础上，为 `d=dv=128`（DeepSeek-R1-Distill-Qwen-1.5B 的真实 head_dim）单独写了两条 tiled + online-softmax 的 flash attention kernel，其余 shape 仍 fallback 到 V1：

- **Prefill**：两级 tiling + online softmax，一次处理整段 prompt。
- **Decode**：一个 warp 处理一个 head，扫全部 KV cache。

`test/benchmark_infer.py` 实测（同一 prompt，`max_steps=64`）：

| 阶段 | V1（全程） | Flash Attention | 加速比 |
| --- | --- | --- | --- |
| Prefill | 1068.98 ms | 520.77 ms | 2.05x |
| Decode | 19.99 ms/token | 16.49 ms/token | 1.21x |

Decode 加速比明显小于 prefill：decode kernel 只发射 `nhead`（12）个 block、每个 1 个 warp，对这块 GPU 的 46 个 SM 来说远未打满，大部分 SM 在 decode 阶段闲置。

### 5.3 Decode Split-KV（flash-decoding），解决 SM 占用不足

针对 5.2 里发现的 decode 占用率问题，把 KV 方向切成多段并行处理（而非 query 行方向，因为 decode 时 `seqlen=1` 没有行可切）：

- **Phase 1**：一个 warp 处理一个 `(head, split)`，在自己负责的 `[split_start, split_end)` 区间内做局部 online softmax，把 `(m, l, acc)` 写入中间 buffer。
- **Phase 2**：一个 warp 处理一个 head，按 `exp(m_split - m_final)` 重新缩放并合并所有 split 的局部结果。
- `num_splits` 按 `nhead * num_splits ≈ 64`（目标 block 数覆盖常见 GPU 的 SM 数量级）动态选取，且不低于 `TILE_K=32` 一个 tile 的量。
- `op.cpp` 按 `total_len > 256` 切换到 split-KV，否则用原 decode kernel——阈值选取见下方分析。

**Kernel 级 microbenchmark**（bf16，真实模型 shape `nhead=12, nkvhead=2, d=dv=128`）：

| total_len | 原 decode kernel | Split-KV | 加速比 |
| --- | --- | --- | --- |
| 128 | 0.058 ms | 0.084 ms | 0.69x（更慢） |
| 256 | 0.110 ms | 0.092 ms | 1.20x |
| 512 | 0.213 ms | 0.096 ms | 2.23x |
| 1000 | 0.443 ms | 0.141 ms | 3.14x |
| 4000 | 2.362 ms | 0.468 ms | 5.05x |
| 16000 | 11.463 ms | 1.567 ms | 7.31x |

`total_len` 越长优势越大：原 kernel 单 warp 串行扫描、耗时线性增长；split-KV 的 block 数随 `total_len` 一起涨，SM 占用率保持住，增长明显更平缓。`total_len=128` 时 split-KV 反而更慢——多一次 kernel launch、多一趟中间结果的显存读写、外加当前实现每次调用都现分配/释放三个中间 buffer 的固定开销，在计算量本身很小时盖不过收益，这正是选 256 作为切换阈值的依据。

端到端 A/B（真实模型，`total_len` 从 67 涨到 467 的一次生成，两种模式各跑一遍）：split-KV 开启 17.41 ms/token（57.45 tok/s）vs 关闭 22.42 ms/token（44.60 tok/s），**约 1.29x**。低于 kernel 级数字是因为这次测量本身跨越了阈值前后两种状态，被平均拉低。

## 6. 支持平台与状态

| 平台 | 算子正确性 | 端到端推理 | 性能 |
| --- | --- | --- | --- |
| CPU | ✅ | ✅ | 未作为优化目标 |
| NVIDIA | ✅ 8/8 | ✅ | cuDNN 移除 + flash attention + split-KV，decode 相比最初的 cuDNN 版本快一个数量级以上 |
| Iluvatar CoreX | ✅ 8/8 | ✅ | 未测量（split-KV 等 NVIDIA 侧优化未移植到这个平台） |
