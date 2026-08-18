# Assignment #4:CUDA 集成 —— 任务说明

> 本文梳理 Assignment #4(`README.md` "Integrate CUDA into LLAISYS" 一节)要完成的工作,对照当前仓库里已有的骨架代码整理成一份可执行的任务清单,写法上跟 [[QWEN2_INFERENCE_ZH]]（`docs/QWEN2_INFERENCE_ZH.md`）保持一致。Assignment #3 已经全部完成并通过验收测试(`test/test_infer.py --test` 输出 `Test passed!`),这是下一阶段的工作。

## 0. 本机环境核对

这台机器(WSL2)实际上已经具备做 CUDA 开发的条件,不用等训练营分配远程资源就能先跑通本地部分:

- `nvidia-smi` 能看到一块 `NVIDIA GeForce RTX 5070 Ti Laptop GPU`。
- `nvcc --version` 是 CUDA 12.9。

也就是说 Nvidia 这一个平台可以完全在本地开发调试。README 要求"从 Nvidia / Iluvatar / Metax / Moore Threads 里选两个平台",第二个平台需要训练营另外批的账号/资源,跟 Nvidia 这部分互不影响,可以先把 Nvidia 走完再申请另一个。

## 1. 目标

给 LLAISYS 加上 CUDA 后端支持,让 Runtime API、所有算子(Assignment #2)、Qwen2 模型推理(Assignment #3)都能在 `LLAISYS_DEVICE_NVIDIA` 上跑,验收标准是:

```bash
python test/test_runtime.py --device nvidia
python test/test_infer.py --model <本地模型目录> --test --device nvidia
```

第二条要求 `--device nvidia` 跑出来的 token 序列跟 HuggingFace CPU/GPU 推理完全一致(逻辑上和 CPU 版本的验收标准相同,只是换了设备)。

## 2. 现状盘点:骨架已经搭到什么程度

跟 Assignment #3 开始时"C 头文件定义好、其余全空"的情况不太一样,这次框架已经把 CUDA 分支的**接口骨架**都占好位了,到处都是 `TO_BE_IMPLEMENTED()`,你要做的是把这些占位填成真正的 CUDA 实现。

### 已经搭好、不用你新建的部分

| 文件/位置 | 现状 |
|---|---|
| `include/llaisys.h` | `LLAISYS_DEVICE_NVIDIA = 1` 已定义 |
| `xmake.lua` | `option("nv-gpu")` 开关、`ENABLE_NVIDIA_API` 宏、`includes("xmake/nvidia.lua")`(条件包含)都已经写好 |
| `src/device/runtime_api.hpp` / `.cpp` | `getRuntimeAPI(device_type)` 的 switch 已经有 `LLAISYS_DEVICE_NVIDIA` 分支,`#ifdef ENABLE_NVIDIA_API` 时会调 `llaisys::device::nvidia::getRuntimeAPI()` |
| `src/device/nvidia/nvidia_runtime_api.cu` | 12 个 Runtime API 函数(`getDeviceCount`/`setDevice`/`deviceSynchronize`/`createStream`/`destroyStream`/`streamSynchronize`/`mallocDevice`/`freeDevice`/`mallocHost`/`freeHost`/`memcpySync`/`memcpyAsync`)**签名都在,函数体全是 `TO_BE_IMPLEMENTED()`** |
| `src/device/nvidia/nvidia_resource.cuh` / `.cu` | `Resource` 类骨架已有(继承 `DeviceResource`),构造函数已经调好基类,目前不需要改 |
| `src/ops/*/op.cpp`(全部 9 个算子:`add`/`argmax`/`embedding`/`linear`/`rearrange`/`rms_norm`/`rope`/`self_attention`/`swiglu`) | 每个的 device switch 里已经有 `#ifdef ENABLE_NVIDIA_API case LLAISYS_DEVICE_NVIDIA: TO_BE_IMPLEMENTED();` 分支占位 |

### 完全是空的、需要你从头写的部分

| 文件 | 现状 |
|---|---|
| `xmake/nvidia.lua` | **不存在**。`xmake.lua` 第 15 行 `includes("xmake/nvidia.lua")` 会直接找不到文件报错——这是你要做的第一件事 |
| `src/ops/*/nvidia/*.cu`(9 个算子各自的 nvidia 子目录) | **不存在**,对照 `src/ops/*/cpu/*.cpp` 的组织方式,每个算子都要补一个 nvidia 版本 |
| `src/llaisys/models/qwen2.cc` 里对多设备的支持 | 目前的 `Create` 只按 `device`/`device_ids` 存了下来,但内部分配张量、KV Cache 时要确认走的是设备无关的路径(理论上应该是,因为都是通过 `Tensor`/`tensor_t` 的构造函数走 `core::context()` 分配,不需要在模型层写 if-cpu-else-nvidia) |

## 3. 要做的工作,按 README 给的顺序拆解

### 3.1 先把编译打通:`xmake/nvidia.lua`

参考 `xmake/cpu.lua` 的写法(`llaisys-device-cpu` 和 `llaisys-ops-cpu` 两个 target),照着建两个对应的 CUDA target:

- `llaisys-device-nvidia`:编译 `src/device/nvidia/*.cu`,注意 `.cu` 文件要用 CUDA 规则编译(xmake 里一般是 `add_rules("cuda")` 或直接靠文件后缀识别,具体语法查 xmake 官方 CUDA 支持文档),还需要设置 CUDA 架构(`add_cugencodes` 或类似 API,对应这块 RTX 5070 Ti 的 compute capability)。
- `llaisys-ops-nvidia`:编译 `src/ops/*/nvidia/*.cu`(通配符路径参照 `cpu.lua` 里 `"../src/ops/*/cpu/*.cpp"` 的写法)。

然后回到根 `xmake.lua`,在 `llaisys-device` 和 `llaisys-ops` 两个 target 里,仿照现有 `add_deps("llaisys-device-cpu")` / `add_deps("llaisys-ops-cpu")`,在 `has_config("nv-gpu")` 时额外 `add_deps("llaisys-device-nvidia")` / `add_deps("llaisys-ops-nvidia")`。

打通后应该能跑:

```bash
xmake f --nv-gpu=y -cv
xmake
xmake install
```

编译不报错(哪怕运行时全是 `TO_BE_IMPLEMENTED()` 抛异常)就算这一步完成。

### 3.2 Runtime API:`src/device/nvidia/nvidia_runtime_api.cu`

对照 `src/device/cpu/cpu_runtime_api.cpp` 那份已经写好的 CPU 实现,每个函数找 CUDA Runtime API 里的对应物:

| 函数 | 对应的 CUDA API(大致) |
|---|---|
| `getDeviceCount` | `cudaGetDeviceCount` |
| `setDevice` | `cudaSetDevice` |
| `deviceSynchronize` | `cudaDeviceSynchronize` |
| `createStream` / `destroyStream` | `cudaStreamCreate` / `cudaStreamDestroy` |
| `streamSynchronize` | `cudaStreamSynchronize` |
| `mallocDevice` / `freeDevice` | `cudaMalloc` / `cudaFree` |
| `mallocHost` / `freeHost` | `cudaMallocHost` / `cudaFreeHost`(pinned memory,不是普通 `malloc`) |
| `memcpySync` | `cudaMemcpy`,注意 `llaisysMemcpyKind_t` 要映射到 `cudaMemcpyKind`(H2D/D2H/D2D 等) |
| `memcpyAsync` | `cudaMemcpyAsync`,多一个 `stream` 参数 |

写完后跑:

```bash
python test/test_runtime.py --device nvidia
```

这个测试逻辑很直接(见 `test/test_runtime.py`):分配两块 device 内存,做 H2D → D2D → D2H 三次拷贝,最后用 `torch.testing.assert_close` 比较,只验证内存管理和拷贝对不对,不涉及计算。

### 3.3 CUDA 算子:`src/ops/*/nvidia/*.cu`

9 个算子(`add`/`argmax`/`embedding`/`linear`/`rearrange`/`rms_norm`/`rope`/`self_attention`/`swiglu`)逐个补齐。`rearrange` 的 CPU 版本本身也还是 `TO_BE_IMPLEMENTED()`(见 [[QWEN2_INFERENCE_ZH]] 里的说明,Assignment #3 靠 `memcpy` 绕过了它),可以放到最后再做,优先级最低。

每个算子的套路是一致的,可以参考同目录下 `cpu/*.cpp` 的实现思路搬到 CUDA kernel 上:

1. 在 `src/ops/<name>/nvidia/` 下新建 `.cuh`(声明)+ `.cu`(kernel 实现),函数签名对照 `src/ops/<name>/cpu/<name>_cpu.hpp` 抄一份。
2. 在 `src/ops/<name>/op.cpp` 里把 `#ifdef ENABLE_NVIDIA_API case LLAISYS_DEVICE_NVIDIA: TO_BE_IMPLEMENTED();` 换成实际调用(参照同一个文件里 `case LLAISYS_DEVICE_CPU` 那一行怎么调 `cpu::xxx(...)`)。
3. `src/ops/<name>/CMakeLists`-等价物这里是靠 `xmake/nvidia.lua` 里 `"../src/ops/*/nvidia/*.cu"` 的通配符自动纳入,不需要逐个算子改 xmake 配置。

建议顺序:先做 `add`(最简单,纯逐元素操作,用来验证整条编译+调用链路通不通),再做 `embedding`/`rms_norm`/`swiglu`/`argmax` 这几个逐元素或简单归约的算子,最后做 `linear`/`rope`/`self_attention` 这几个涉及矩阵乘法或者复杂索引的。`linear`(矩阵乘法)可以考虑直接用 cuBLAS(`cublasSgemm`/`cublasGemmEx` 等,注意这个模型权重是 BF16),不用手写 kernel。

每实现完一个算子,可以用 Assignment #2 现成的算子测试脚本加 `--device nvidia` 跑(`test/ops/` 目录下,具体参照 `test/ops/*.py` 里已有的 CPU 用例怎么写,应该有 `--device` 参数)。

### 3.4 把 Qwen2 模型接到 CUDA

理论上如果 3.1~3.3 做完了,`src/llaisys/models/qwen2.cc` 不需要额外改动——你在 Assignment #3 里写的 `Infer` 全部是通过 `ops::xxx(...)` 调用算子、通过 `Tensor` 构造函数分配张量,只要这些底层调用是设备无关的(靠 `tensor->deviceType()` 在 op.cpp 内部分发),模型层的 C++ 代码本身不用感知 CPU/NVIDIA 的区别。

需要确认(不代表一定要改,先去看代码是否已经这样写的):
- `LlaisysQwen2ModelCreate` 里分配权重张量、KV Cache 张量时用的 `device`/`device_ids` 参数,有没有正确传给 `Tensor` 的构造(而不是被忽略、默认写死成 CPU)。
- Python 侧 `python/llaisys/models/qwen2.py` 的 `__init__` 接的 `device: DeviceType` 参数有没有真的传下去(现在的实现已经支持传参,回顾 `LIB_LLAISYS.llaisysQwen2ModelCreate(ctypes.byref(self.meta), device.value, device_ids, 1)` 这一行)。

验收:

```bash
python test/test_infer.py --model <本地模型目录> --test --device nvidia
```

## 4. 调试建议

- 跟 Assignment #3 一样,先用最小规模验证:单个算子(比如 `add`)先跑通,再逐步加算子。
- CUDA kernel 写错很容易表现为"编译通过但结果不对"或者直接 core dump,建议开发时先在小 shape、固定输入下用 `cuda-memcheck` / `compute-sanitizer` 排查越界访问,比对着 CPU 版本结果肉眼找 bug 快得多。
- BF16 在 CUDA 上要用 `__nv_bfloat16` / `<cuda_bf16.h>`,跟 CPU 端用 `ml_dtypes`/内存直接 memcpy 的方式不一样,注意类型转换。
- `self_attention` 是最复杂的一个,可以先在 CPU 上把结果存下来(用 `tensor.debug()`),CUDA 版本跑出来后逐值比对。

## 5. 验收

```bash
python test/test_runtime.py --device nvidia
python test/test_infer.py --model <本地模型目录> --test --device nvidia
```

两条都通过、`Test passed!` 打印出来即算完成。之后按 README 说明 commit + push,CI 里 Assignment #4 对应步骤应该跑绿(注意 CI runner 大概率没有 GPU,这部分很可能是训练营额外配置的自跑或人工验收环节,具体以 README "Assignment Submission Requirements" 一节的最新说明为准)。
