# LLAISYS Assignment #4 报告

CUDA 集成 + 双平台适配（NVIDIA、天数智芯 Iluvatar CoreX），以及后续的正确性修复与性能分析。

## 1. 环境

**NVIDIA（本地开发机，WSL2）**
- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU（`sm_120`，Blackwell 架构）
- CUDA：12.9（`nvcc release 12.9, V12.9.41`）
- cuDNN：9.20.0（曾升级到 9.24.0 用于 RoPE 的 Graph API 探索，但发现 9.24.0 的 SDPA 执行引擎在 `sm_120` 上运行时崩溃——`CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER`，与 shape/GQA/causal 设置无关，是 cuDNN 该版本本身在这块 GPU 上的兼容性问题。已回退到 9.20.0，全部 8 个算子测试通过，无回归）

**天数智芯 Iluvatar CoreX（远程云平台，仅能通过生成诊断脚本远程执行 + 用户回传结果的方式协作，无直接 SSH）**
- 硬件：Iluvatar BI-V150
- SDK：`corex-4.4.0`
- 编译器：`/usr/local/corex/bin/clang++`（LLVM/clang 18，用 `-x ivcore` 而非标准 `-x cuda`）
- cuBLAS：`libcublas.so.10.2.3.254`（CUDA 10.2 时代的兼容库）
- cuDNN：7.6.5（早于 Graph API 存在的版本，因此 `self_attention` 的 cuDNN SDPA 路径在这个平台上会自动 fallback 到 V1 手写 kernel）

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
| self_attention | ✅ 通过（cuDNN SDPA + V1 fallback） | ✅ 通过（V1 fallback，见下方说明） |
| rearrange | 不实现（作业 #2 遗留的废弃算子，永久 stub） | 同左 |

**Iluvatar `linear` 的 bf16 插曲**：一开始 bf16 分支直接照抄 NVIDIA 那边用 `cublasSgemmEx`，数值算错了。排查后确认 Iluvatar 这个 CUDA-10.2 时代的 cuBLAS 兼容库里，`cublasSgemmEx` 对 `CUDA_R_16BF` 返回 `CUBLAS_STATUS_NOT_SUPPORTED`（15）——这个函数在这个版本压根不支持 bf16。翻头文件发现虽然报的是 10.2 版本号，但头文件里还是带了 `cublasComputeType_t`、`CUBLAS_COMPUTE_32F_FAST_16BF` 这些更新的概念，说明底层库做过修改。改用另一个更通用的函数 `cublasGemmEx`（注意 `computeType` 参数在这份头文件里仍是老式的 `cudaDataType` 类型，要传 `CUDA_R_32F` 而不是 `CUBLAS_COMPUTE_32F`）后，bf16 数值完全正确，所有 shape 都通过。

**Iluvatar `self_attention` 为什么直接能用 V1 fallback**：这个平台的 cuDNN 是 7.6.5，早于 Graph API（cuDNN 8+ 才有）存在的年代。`self_attention_cuda.cu` 里本来就有一个 `#if CUDNN_MAJOR >= 8` 的编译期 guard（当初是为了应对本机 sm_120 的 cuDNN 9.24.0 运行时问题加的），在 Iluvatar 上这个 guard 天然跳过了 `cudnn_frontend`/SDPA 代码块，直接走 V1 手写 kernel——不需要任何 Iluvatar 专属改动。

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

**下一步（尚未实现）**：把已经 build 好的图按 shape（`seqlen, total_len, nhead, nkvhead, d, dv`）缓存起来，decode 阶段除了 `total_len` 逐步 +1 之外其他维度都不变，命中缓存时直接复用，不用每次都重新走一遍建图流程。

### Profiling 工具踩坑记录

- `nsys`（Nsight Systems）能直接用，但发现一个真实的坑：给 `self_attention.py --device nvidia` 套上 `nsys profile` 之后，GQA 那个测试 case 会报 `[cudnn_frontend] build_plans failed: No valid execution plans built.`——不加 profiler 就是好的，复现了两次。说明 profiling 工具本身会干扰 cuDNN 的运行时 heuristic 选择。
- `ncu`（Nsight Compute）需要管理员权限读取 GPU 性能计数器（`ERR_NVGPUCTRPERM`），WSL2 上很常见，需要交互式 `sudo`。
- 完整 kernel 级别的 `nsys stats --report cuda_gpu_kern_sum` 这次没有直接拿到数据（cuBLAS 内部可能是走 driver API 而非 runtime API 发起 kernel launch，需要调整 trace 参数）。
- 代码里目前完全没有 NVTX 标注；`test/benchmark_infer.py` 加了一个可选的 `--nvtx` 参数（配合 `qwen2.py` 新增的 `step_context` 钩子），可以在 nsys 时间线里标出 prefill/decode 每一步的范围。

## 6. 支持平台与状态

| 平台 | 算子正确性 | 端到端推理 | 性能 |
| --- | --- | --- | --- |
| CPU | ✅ | ✅ | 未作为优化目标 |
| NVIDIA | ✅ 8/8 | ✅（修复段错误后） | 已定位瓶颈（`self_attention` 建图开销），优化尚未实现 |
| Iluvatar CoreX | ✅ 8/8 | 未验证 | 未测量 |
