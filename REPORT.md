# LLAISYS Assignment #4 报告

CUDA 集成 + 双平台适配（NVIDIA、天数智芯 Iluvatar CoreX），以及后续的正确性修复与性能分析。

## 1. 环境

**NVIDIA（本地开发机，WSL2）**
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU（`sm_120`，Blackwell 架构）
- CUDA：12.9（`nvcc release 12.9, V12.9.41`）
- cuDNN：曾经用过（`self_attention` 的 SDPA Graph API 加速路径，见第 5 节），但已经在评估后决定去掉——多版本兼容性问题（9.24.0 的 SDPA 执行引擎在这块 `sm_120` GPU 上运行时崩溃）、Iluvatar 平台完全没有 Graph API、以及缓存方案本身要正确处理 padding+causal mask 的复杂度,综合下来认为不适合这个项目现阶段。现在项目不再依赖 cuDNN。

**天数智芯 Iluvatar CoreX（远程云平台，仅能通过生成诊断脚本远程执行 + 用户回传结果的方式协作，无直接 SSH）**
- 硬件：Iluvatar BI-V150
- SDK：`corex-4.4.0`
- 编译器：`/usr/local/corex/bin/clang++`（LLVM/clang 18，用 `-x ivcore` 而非标准 `-x cuda`）
- cuBLAS：`libcublas.so.10.2.3.254`（CUDA 10.2 时代的兼容库）
- cuDNN：7.6.5（早于 Graph API 存在的版本；不过 `self_attention` 现在已经不用 cuDNN 了，这一点不再影响它）

## 2. 复现步骤

**NVIDIA：**

```bash
xmake f --nv-gpu=y
xmake
xmake install
pip install -e ./python

python test/ops/<op>.py --device nvidia
python test/test_infer.py --model <dir_path/to/model> --test --device nvidia
```

**Iluvatar（在远程机器上）：**

```bash
export XMAKE_ROOT=y   # 容器内是 root 用户，xmake 默认拒绝 root 运行

xmake f --iluvatar-gpu=y
xmake
xmake install
pip install -e ./python

python test/ops/<op>.py --device iluvatar
```

远程机器上没有预装 `xmake`，需要先 `curl -fsSL https://xmake.io/shget.text | bash` 安装（注意不要带 `--branch` 参数，装脚本的参数解析在这个 flag 上有 bug）。

## 3. 逐算子结果

| 算子 | NVIDIA | Iluvatar |
| --- | --- | --- |
| add | ✅ 通过 | ✅ 通过 |
| embedding | ✅ 通过 | ✅ 通过 |
| argmax | ✅ 通过 | ✅ 通过 |
| rope | ✅ 通过 | ✅ 通过 |
| linear | ✅ 通过（cuBLAS 后端） | ✅ 通过（cuBLAS 兼容库后端，见下方 bf16 说明） |
| swiglu | ✅ 通过 | ✅ 通过 |
| rms_norm | ✅ 通过 | ✅ 通过 |
| self_attention | ✅ 通过（V1 手写 kernel，见下方说明） | ✅ 通过（同一份 V1 手写 kernel） |
| rearrange | 不实现（作业 #2 遗留的废弃算子，永久 stub） | 同左 |

**Iluvatar `linear` 的 bf16 插曲**：一开始 bf16 分支直接照抄 NVIDIA 那边用 `cublasSgemmEx`，数值算错了。排查后确认 Iluvatar 这个 CUDA-10.2 时代的 cuBLAS 兼容库里，`cublasSgemmEx` 对 `CUDA_R_16BF` 返回 `CUBLAS_STATUS_NOT_SUPPORTED`（15）——这个函数在这个版本压根不支持 bf16。翻头文件发现虽然报的是 10.2 版本号，但头文件里还是带了 `cublasComputeType_t`、`CUBLAS_COMPUTE_32F_FAST_16BF` 这些更新的概念，说明底层库做过修改。改用另一个更通用的函数 `cublasGemmEx`（注意 `computeType` 参数在这份头文件里仍是老式的 `cudaDataType` 类型，要传 `CUDA_R_32F` 而不是 `CUBLAS_COMPUTE_32F`）后，bf16 数值完全正确，所有 shape 都通过。

**`self_attention` 为什么两个平台都是 V1 kernel**：`self_attention` 原来在 NVIDIA 上有一条用 cuDNN `cudnn_frontend` SDPA Graph API 加速的路径（head 维度是 8 的倍数时走这条路），Iluvatar 上因为它的 cuDNN 只有 7.6.5（早于 Graph API 存在的年代）天然走不了这条路，一直是靠 V1 手写 kernel fallback。后来评估后决定把 cuDNN 加速这条路整体去掉（详见第 5 节），现在两个平台统一用同一份 V1 手写 kernel，不存在 fallback 概念了。

## 4. 完整端到端推理（`test_infer.py --test`）

| 平台 | 状态 |
| --- | --- |
| CPU | ✅ 通过 |
| NVIDIA | ✅ 通过（见下方"发现并修复的问题"） |
| Iluvatar | 尚未验证（8 个算子已全部通过，完整模型推理还没在远程机器上跑过） |

### 发现并修复的问题：`qwen2.cc` 在 NVIDIA 上的段错误

在准备做性能分析、第一次真正跑通 `test_infer.py --device nvidia` 时（`self_attention`/`linear` 换成 cuDNN/cuBLAS 之后，完整模型推理其实从没被重新验证过），发现直接段错误。

用 `git stash` 排除了是本阶段新改动导致的，再用 debug 模式编译 + `gdb` 抓到了确切位置：`src/llaisys/models/qwen2.cc:251`

```cpp
return *reinterpret_cast<int64_t*>(max_idx->data());
```

`max_idx` 是在 GPU 设备上创建的 tensor，`->data()` 返回的是**显存地址**，直接在 CPU 端解引用显存指针是非法内存访问。这个 bug 之前一直没被发现，是因为完整推理只在 CPU 设备上验证过——CPU 设备下"显存指针"和主机指针是同一个地址空间，凑巧不会崩。

修复方式参考了项目里已有的 `Tensor::debug()`（`src/tensor/tensor.cpp:149-164`）读取 GPU 数据的写法：非 CPU 设备时，用 `memcpy_sync(..., LLAISYS_MEMCPY_D2H)` 把这一个 int64 值拷贝回主机端变量，再返回，而不是直接解引用显存指针。

修复后 `test_infer.py --model DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --test` 完整通过，32 个 token 逐个匹配 HF 参考实现。

## 5. 性能分析（进行中）

修复正确性问题后，第一次拿到了真实、可信的端到端耗时数据：

| | HF (PyTorch) | llaisys |
| --- | --- | --- |
| 32 个 token 总耗时 | 5.59s | 27.28s |

llaisys 目前比 HF 慢约 5 倍。用新写的 `test/benchmark_infer.py`（把 prefill 和 decode 阶段分开计时）测得：

| 阶段 | 耗时 |
| --- | --- |
| Prefill（9 个 prompt token） | 1552.57 ms |
| Decode（31 步） | 平均 640.06 ms/token（1.56 tokens/s） |

**Decode 阶段是绝对的瓶颈**（31 步就占了总耗时的绝大部分）。进一步用真实 decode shape（`qlen=1, kvlen=20, nh=12, nkvh=2, dh=128`，对应 DeepSeek-R1-Distill-Qwen-1.5B 的真实配置）做单算子微基准，定位到：

| 算子 | 耗时 |
| --- | --- |
| `self_attention`（decode shape） | **20.36 ms/call** |
| `linear`（decode shape，qkv/o-proj，`M=1,K=1536,N=1536`） | 0.012 ms/call |
| `linear`（decode shape，mlp gate/up，`M=1,K=1536,N=8960`） | 0.039 ms/call |

模型有 28 层，每层调一次 `self_attention`：28 × 20.36ms ≈ 570ms，和实测的 640ms/token 基本吻合。**`linear`（已经换成 cuBLAS）完全不是问题，瓶颈几乎全在 `self_attention`。**

根因：`self_attention_cuda.cu` 里 cuDNN 的 SDPA 图（`cudnn_frontend::graph::Graph`）**每次调用都从头构建**（`validate → build_operation_graph → create_execution_plans → build_plans`），代码里早就留了这条 TODO 但一直没验证影响有多大。对 `qlen=1, kvlen=20` 这么小的计算量，真正的矩阵运算应该是微秒级，20ms 基本全是建图开销，不是算力开销。

**关键设计难点**：decode 阶段 `total_len`（KV-cache 已有长度）每一步都 +1，而 K/V 的 `Tensor_attributes` 把 `total_len` 直接焊进了 graph 的静态维度里，按精确 shape 做缓存永远不会命中。研究过几条给 graph 加缓存的路：cuDNN Graph API 自己的动态 shape 机制（`set_dynamic_shape_enabled`）被 SDPA 的 "Unified" 引擎明确拒绝；cuDNN 专为这个场景设计的 ragged/变长 KV 机制（`set_seq_len_q`/`set_seq_len_kv`）在这个 cudnn_frontend 版本里被注释掉、没有开放；剩下唯一可行的分桶方案，又要求每次调用把 K/V pad 到桶大小，而 `causal_mask_bottom_right` 会把 padding 出来的 key 当成真实的靠后 token 参与因果对齐——不加额外的显式 mask 会**算出错误结果**，这部分怎么做没有设计清楚。

**最终决定（2026-08-06）：去掉 cuDNN 加速，`self_attention` 统一改回 V1 手写 kernel。** 除了上面这个缓存设计本身没解决之外，cuDNN 这条路整体上还有两个不利因素：一是版本/平台兼容性脆弱——本机 cuDNN 9.24.0 的 SDPA 执行引擎在 `sm_120` GPU 上运行时直接崩溃，Iluvatar 平台的 cuDNN 7.6.5 更是完全没有 Graph API；二是真实生产推理引擎（vLLM、TensorRT-LLM）的核心 attention kernel 本来就是手写/模板 kernel，`seqlen`/`total_len` 是普通运行时参数、不是编译期焊死的常量，根本不存在"建图"这个开销——这和 llaisys 自己的 V1 手写 kernel 是同一种设计哲学。综合评估后认为 cuDNN 加速这条路对这个教学项目现阶段不合适，已经从 `self_attention` 的 NVIDIA、Iluvatar 两份实现里彻底去掉（不再链接 `cudnn`/`nvrtc`，device Resource 也不再持有 `cudnnHandle_t`）。

**去掉之后重新测的数据（`test/benchmark_infer.py --device nvidia`）**：

| 阶段 | cuDNN 版本（旧） | V1 kernel（现在） |
| --- | --- | --- |
| Decode | 640.06 ms/token（1.56 tokens/s） | **78.37 ms/token（12.76 tokens/s）** |

decode 延迟降到约 1/8，直接印证了"瓶颈是建图开销、不是算力"这个判断——去掉建图开销后，V1 手写 kernel 在这个 decode shape（`qlen=1`）下反而明显更快。`test/ops/self_attention.py --device nvidia` 和 `test/test_infer.py --device nvidia --test` 都重新跑过，正确性无回归。

### Profiling 工具踩坑记录

- `nsys`（Nsight Systems）能直接用，做 CUDA API 级别的 trace 没问题。（写这条笔记的时候 `self_attention` 还在用 cuDNN 的 SDPA 图，当时发现给它套上 `nsys profile` 会让 GQA 测试 case 报 `build_plans failed`——profiling 工具会干扰 cuDNN 的运行时 heuristic 选择；`self_attention` 现在已经不用 cuDNN 了，这个坑不再适用。）
- `ncu`（Nsight Compute）需要管理员权限读取 GPU 性能计数器（`ERR_NVGPUCTRPERM`），WSL2 上很常见，需要交互式 `sudo`。
- 完整 kernel 级别的 `nsys stats --report cuda_gpu_kern_sum` 这次没有直接拿到数据（cuBLAS 内部可能是走 driver API 而非 runtime API 发起 kernel launch，需要调整 trace 参数）。
- 代码里目前完全没有 NVTX 标注；`test/benchmark_infer.py` 加了一个可选的 `--nvtx` 参数（配合 `qwen2.py` 新增的 `step_context` 钩子），可以在 nsys 时间线里标出 prefill/decode 每一步的范围。

## 6. 支持平台与状态

| 平台 | 算子正确性 | 端到端推理 | 性能 |
| --- | --- | --- | --- |
| CPU | ✅ | ✅ | 未作为优化目标 |
| NVIDIA | ✅ 8/8 | ✅（修复段错误后） | 定位到瓶颈（`self_attention` cuDNN 建图开销）后去掉了 cuDNN，decode 从 640ms/token 降到 78ms/token |
| Iluvatar CoreX | ✅ 8/8 | 未验证 | 未测量 |

## 7. 复测记录（2026-08-06）

在开始 `self_attention` 的 flash attention 手写重写（`src/ops/self_attention/nvidia/flash_attention_cuda.cu`，仍在开发中，只做完了阶段 1 的 warp-shuffle 规约）之前，先把项目在 CPU 和本机 NVIDIA GPU 上完整跑了一遍，确认当前稳定状态没问题。测试前把 `op.cpp` 里临时指向 `flash_attention`（为了跑 `test/ops/self_attention.py` 验证阶段 1）的改动改回了 `self_attention`，所以下面结果反映的是项目当前稳定状态，不包含还没做完的 flash attention 工作。

**8 个算子 + 端到端推理**：

| 平台 | 8 个算子 | 端到端推理正确性 |
| --- | --- | --- |
| CPU | ✅ 8/8 通过 | ✅ 通过，25 个 token 逐个匹配 HF |
| NVIDIA | ✅ 8/8 通过 | ✅ 通过，25 个 token 逐个匹配 HF |

两边全绿，没有回归。

**性能（NVIDIA，`benchmark_infer.py`）**：

| 阶段 | 耗时 |
| --- | --- |
| Prefill（9 个 prompt token） | 618.65 ms |
| Decode（80 步） | 平均 13.81 ms/token（72.4 tokens/s） |

比第 5 节记的"去掉 cuDNN 后 78.37ms/token"又快了不少，大概率是那次测的是冷启动（CUDA context 初始化、cuBLAS 首次调用的 JIT 开销都算进了第一次调用），这次是热身状态下测的，两次不是同一口径，不代表这段时间又做了什么优化——如果要拿准确数字做前后对比，需要固定一套预热流程再测。

**构建**：CPU-only（`xmake f -c`）和 NVIDIA（`xmake f -c --nv-gpu=y`）两种配置都是干净编译；NVIDIA 构建确认没有链接 cuDNN，符合第 5 节里去掉 cuDNN 加速的决定。

结论：项目当前稳定状态在 CPU 和本地 GPU 上都是健康的，可以放心在这个基础上继续 flash attention 的开发。

**天数（Iluvatar）平台的 cuDNN 清理复查**：去掉 cuDNN 加速（第 5 节）那次改动是 NVIDIA、Iluvatar 两份 `self_attention` 实现一起改的，这次专门复查了一遍 Iluvatar 那边有没有清理干净：

- `src/ops/self_attention/iluvatar/self_attention_iluvatar.cu`/`.cuh`：没有 `#include <cudnn.h>`，没有 `cudnn_frontend`，没有 `resource` 参数——跟 NVIDIA 那份一样，现在是纯 V1 手写 kernel。
- `src/device/iluvatar/iluvatar_resource.cu`/`.cuh`：`cudnnHandle_t`、`cudnnCreate`/`cudnnDestroy` 都删掉了，只留 `cublasHandle_t`（`linear` 还要用）。
- `xmake/iluvatar.lua`：两个 target 都不再 `add_links("cudnn")`。
- 唯一还出现"cudnn"字样的地方是 `self_attention_iluvatar.cu` 里解释"为什么去掉 cuDNN"的注释，不是代码；`rope_iluvatar.cu`、`argmax_iluvatar.cu` 里各有一行提到 cudnn 的注释，是这两个算子自己的历史 detour 记录，跟这次的 `self_attention` 清理无关。

这个复查是看代码本身确认的（不再引用任何 cudnn 头文件/符号/链接），逻辑上没问题，但**没有在天数那台远程机器上实际跑 `xmake build --iluvatar-gpu=y` 重新编译验证**——本机没有天数的工具链，这一步还需要在远程机器上补跑一次才能完全确认。
