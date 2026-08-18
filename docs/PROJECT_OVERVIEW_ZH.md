# LLAISYS 项目总览

## 1. 项目定位

LLAISYS（Let's Learn AI SYStem）是一个面向 AI 系统初学者的教学项目，目标是让学习者从底层开始实现一个小型推理系统，而不是直接调用成熟框架完成计算。

项目采用分层设计：C++17 实现运行时、内存管理、张量和算子，通过稳定的 C ABI 导出共享库；Python 使用 `ctypes` 调用该共享库，并借助 PyTorch 生成参考结果和验证数值正确性。课程最终目标是使用这些基础模块加载 Qwen2 架构模型，并复现 DeepSeek-R1-Distill-Qwen-1.5B 的推理结果。完成作业阶段并通过考核的学员，还可以进入项目阶段，为正式的 InfiniLM 推理引擎贡献代码。

> 本文描述的是当前仓库快照（已核对 `src/`、`include/`、`xmake.lua` 源码）。该仓库本身是一套待完成的课程骨架，很多函数故意保留为 `TO_BE_IMPLEMENTED()`，不应把所有已声明接口理解为已经可用。

## 2. 整体架构

一次典型调用会经过以下链路：

```text
用户 / Python 测试
        │
        ▼
Python 友好接口（python/llaisys/*.py）
        │
        ▼ ctypes
C ABI 绑定（python/llaisys/libllaisys + include/llaisys）
        │
        ▼
C++ 边界层（src/llaisys/*.cc）── unwrap LlaisysTensor -> tensor_t，调用 C++ 实现，再包装返回
        │
        ├── 张量（src/tensor）── TensorMeta + Storage(shared_ptr) + offset
        ├── 算子（src/ops）── 参数校验 -> dtype 分派 -> CPU/NVIDIA kernel
        └── 核心运行时（src/core）── Context（线程局部单例） -> Runtime -> MemoryAllocator
                 │
                 ▼
设备后端（src/device/cpu 完整实现；nvidia 为预留骨架，受 ENABLE_NVIDIA_API 宏保护）
```

这种结构有两个主要教学价值：一是展示 Python 框架接口如何跨语言进入原生后端；二是把与设备无关的张量/算子逻辑和 CPU、GPU 等设备实现分离。

## 3. 目录说明

| 路径 | 作用 |
| --- | --- |
| `include/` | 对外公开的 C API：`llaisys.h`（设备/数据类型枚举）、`llaisys/{runtime,tensor,ops}.h`、`llaisys/models/qwen2.h`。 |
| `src/llaisys/` | C ABI 到内部 C++ 对象的适配边界（`tensor.cc`、`ops.cc`、`runtime.cc`）。 |
| `src/core/` | `context/`、`runtime/`、`storage/`、`allocator/` —— 设备无关的运行时基础设施。 |
| `src/device/` | 设备资源与 `LlaisysRuntimeAPI` 函数表；`cpu/` 是完整实现，`nvidia/` 为预留骨架。 |
| `src/tensor/` | `Tensor`/`TensorMeta` —— 形状、步长、存储引用、视图变换和数据搬运。 |
| `src/ops/` | 每个算子一个子目录：`add`、`argmax`、`embedding`、`linear`、`rms_norm`、`rope`、`self_attention`、`swiglu`、`rearrange`，各自再分 `cpu/`（未来还有 `nvidia/`）。 |
| `src/utils/`、`src/utils.hpp` | `check.hpp`（断言/校验宏）、`types.hpp`/`types.cpp`（dtype 大小、fp16/bf16 转换）。 |
| `python/llaisys/libllaisys/` | 共享库加载和底层 `ctypes` 函数签名，一一对应 `include/llaisys/*.h`。 |
| `python/llaisys/` | `RuntimeAPI`、`Tensor`、`Ops` 等 Python 接口。 |
| `python/llaisys/models/` | 模型层，目前 `qwen2.py` 提供待实现的 Qwen2 骨架（构造函数和 `generate` 均为 TODO）。 |
| `test/` | 运行时、张量、算子和端到端推理测试，以 PyTorch 为参考。 |
| `xmake.lua`、`xmake/` | C++ 构建配置：`llaisys-utils`/`llaisys-device`/`llaisys-core`/`llaisys-tensor`/`llaisys-ops` 静态库 + `llaisys` 共享库；`xmake/cpu.lua` 定义 CPU 子目标。 |
| `.github/workflows/build.yaml` | Windows、Ubuntu 上的构建和分阶段作业测试。 |

## 4. C++ 构建体系与静态库依赖图

`xmake.lua` 把 C++ 代码拆成多个静态库目标，最终一起链接进 `llaisys` 共享库，依赖关系如下：

```text
llaisys-utils  (src/utils/*.cpp：dtype 大小、fp16/bf16 转换)
     ▲
     │
llaisys-device-cpu (src/device/cpu/*.cpp)  ──┐
     ▲                                       │
     │                                       ▼
llaisys-device  (src/device/*.cpp)  ← 依赖 llaisys-utils + llaisys-device-cpu
     ▲
     │
llaisys-core    (src/core/*/*.cpp)  ← 依赖 llaisys-utils + llaisys-device
     ▲
     │
llaisys-tensor  (src/tensor/*.cpp)  ← 依赖 llaisys-core
     ▲
     │
llaisys-ops-cpu (src/ops/*/cpu/*.cpp) ← 依赖 llaisys-tensor
     ▲
     │
llaisys-ops     (src/ops/*/*.cpp)   ← 依赖 llaisys-ops-cpu
     │
     ▼
llaisys（共享库，src/llaisys/*.cc）← 依赖以上全部
     │
     ▼ xmake install 后自动 os.cp
python/llaisys/libllaisys/*.so（或 Windows 下 *.dll）
```

几个值得注意的构建细节：

- 所有目标统一 `set_languages("cxx17")` 并开启 `set_warnings("all", "error")`（警告即错误），非 Windows 平台加 `-fPIC -Wno-unknown-pragmas`。
- `nv-gpu` 是一个 xmake `option`，默认 `false`；打开后会 `add_defines("ENABLE_NVIDIA_API")` 并 `includes("xmake/nvidia.lua")`——但该文件目前**不存在于仓库**，所以 `--nv-gpu=y` 现在还编译不过，是留给作业 #4 的扩展点。
- `xmake install` 的 `after_install` 钩子会把编译好的 `lib/*.so`（Linux）拷贝到 `python/llaisys/libllaisys/`，这是 Python 侧 `ctypes.CDLL` 能加载到最新库的关键一步——改完 C++ 代码后必须重新 `xmake && xmake install` 才会生效。

## 5. 核心运行时（`src/core/`）详解

### 5.1 Context：线程局部单例

`Context` 通过函数内 `static thread_local` 实现每线程唯一实例：

```cpp
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}
```

构造函数会遍历所有 `llaisysDeviceType_t`（NVIDIA 排在前面，CPU 特意放最后作为兜底），对每种设备调用 `llaisysGetRuntimeAPI(device_type)->get_device_count()` 探测可用设备数，并为每个设备号预留一个 `Runtime*` 槽位（此时先不创建，除非它是第一个被找到的可用设备，那个会被立即创建并激活为 `_current_runtime`）。由于 CPU 后端的 `get_device_count()` 恒为 `1`，NVIDIA 分支在没有实现 GPU Runtime API 时通过 `getUnsupportedRuntimeAPI()` 返回 `get_device_count() == 0`，所以**默认总是落到 CPU**。

`setDevice(device_type, device_id)` 会检查当前激活的 Runtime 是否已经匹配，若不匹配则 `_deactivate()` 旧 Runtime、惰性 `new Runtime(...)`（如果该槽位还没创建过）、再 `_activate()` 新 Runtime。`Context` 禁止拷贝和移动（`= delete`），保证每个线程只有一份状态。

### 5.2 Runtime：单设备资源管理器

`Runtime` 持有一个设备的 `LlaisysRuntimeAPI` 函数表指针、一条 `llaisysStream_t`、一个 `MemoryAllocator*`。构造时通过 `llaisys::device::getRuntimeAPI(device_type)` 拿到函数表，再 `_api->create_stream()` 建流、`new allocators::NaiveAllocator(_api)` 建分配器。

`allocateDeviceStorage(size)` / `allocateHostStorage(size)` 分别调用分配器的 `allocate()` 或运行时 API 的 `malloc_host()`，把裸指针包进一个新建的 `Storage`（`is_host` 标志区分二者）。`freeStorage(Storage*)` 则根据 `storage->isHost()` 决定走 `_api->free_host()` 还是分配器的 `release()`——这个判断逻辑是 `Storage` 析构时自动触发的（见 5.3）。

### 5.3 Storage：跨张量共享的内存所有权

```cpp
class Storage {
    std::byte *_memory; size_t _size; Runtime &_runtime; bool _is_host;
    ~Storage() { _runtime.freeStorage(this); }
};
```

`Storage` 的构造函数是私有的，只有 `Runtime` 是 `friend` 能创建它，这保证了"内存块的生命周期必须由分配它的 Runtime 来管理"这一不变式。`Tensor` 通过 `core::storage_t`（即 `std::shared_ptr<Storage>`）持有它，多个 `Tensor`（例如 `view`/`permute`/`slice` 产生的视图）可以共享同一个 `Storage`，最后一个 `shared_ptr` 析构时才真正释放内存。

### 5.4 MemoryAllocator：抽象分配策略

`MemoryAllocator` 是一个只有两个纯虚函数的接口（`allocate`/`release`），当前唯一实现 `NaiveAllocator` 直接透传给 `_api->malloc_device()` / `_api->free_device()`，不做池化或复用——这是刻意简化的教学版本，如果要做性能优化（作业之外的扩展方向），这里是天然的切入点。

### 5.5 LlaisysRuntimeAPI：设备无关的函数表

`include/llaisys/runtime.h` 定义了一张 12 个函数指针组成的 C 结构体（设备数/切换设备/同步、流的创建销毁同步、设备内存与主机内存的分配释放、同步/异步拷贝）。`src/device/runtime_api.cpp` 里的 `getRuntimeAPI(device_type)` 按 `device_type` 分发：

- `LLAISYS_DEVICE_CPU` → `llaisys::device::cpu::getRuntimeAPI()`（`src/device/cpu/cpu_runtime_api.cpp`，全部基于 `std::malloc/std::free/std::memcpy` 实现，"设备内存"和"主机内存"其实是同一块主机内存，`memcpySync`/`Async` 不区分 `kind` 直接 `memcpy`——这让 CPU 后端非常适合先验证跨设备抽象是否设计正确，再去接入真正异构的 NVIDIA 后端）；
- `LLAISYS_DEVICE_NVIDIA` → 若定义了 `ENABLE_NVIDIA_API` 则调用 `nvidia::getRuntimeAPI()`（目前仓库中**没有对应源文件**，只在头文件里声明了函数原型），否则退回 `getUnsupportedRuntimeAPI()`——这是一张所有函数都直接 `throw std::runtime_error` 的"空实现"函数表，用于在未启用 NVIDIA 支持时给出明确报错而不是链接失败。

## 6. 张量实现（`src/tensor/tensor.hpp` / `tensor.cpp`）

### 6.1 数据结构

```cpp
struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
};

class Tensor {
    TensorMeta _meta;
    core::storage_t _storage;   // shared_ptr<core::Storage>
    size_t _offset;              // 相对 Storage 起点的字节偏移
};
```

`Tensor` 的构造函数是私有的，只能通过静态工厂 `Tensor::create(shape, dtype, device_type, device)` 或内部的 `new Tensor(meta, storage, offset)` 创建，返回类型统一是 `tensor_t = std::shared_ptr<Tensor>`。

### 6.2 `Tensor::create` 的设备放置逻辑

```cpp
if (device_type == LLAISYS_DEVICE_CPU
    && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
    // 当前激活的是非 CPU 设备，但要建的张量指定为 CPU ——
    // 分配"主机存储"（pinned/host memory），不改变当前激活设备
    auto storage = core::context().runtime().allocateHostStorage(bytes);
} else {
    // 目标设备就是当前设备，或本来就要切设备：先 setDevice 再分配设备存储
    core::context().setDevice(device_type, device);
    auto storage = core::context().runtime().allocateDeviceStorage(bytes);
}
```

新建张量默认按行主序（C-contiguous）计算 `strides`：从最后一维往前累乘 `shape`，这也是 `numel()`（`std::accumulate` 对 `shape` 做乘积）和 `elementSize()`（`utils::dsize(dtype)`）的基础。

### 6.3 `debug()` 与模板化打印

`info()` 输出形状/步长/dtype 的字符串摘要；`debug()` 先做 `device_synchronize()`，若张量在设备上则先 `memcpy_sync(..., D2H)` 拷到一块临时 CPU 张量，再调用按 dtype 分派的 `debug_print<T>`（`print_data` 模板函数按 stride 递归打印每一维，`fp16_t`/`bf16_t` 会先 `utils::cast<float>` 再打印）。这个函数是调试作业实现时对拍 PyTorch 张量数值的主要工具。

### 6.4 已实现 vs 待实现的方法（按源码逐一核对）

| 方法 | 状态 | 说明 |
| --- | --- | --- |
| `create` / `data` / `ndim` / `shape` / `strides` / `dtype` / `deviceType` / `deviceId` / `numel` / `elementSize` | ✅ 已实现 | 元信息查询与工厂函数，作业 #1 之前就已提供。 |
| `info()` / `debug()` | ✅ 已实现 | 调试打印，覆盖全部 dtype。 |
| `isContiguous()`（`tensor.cpp:166`） | ❌ `TO_BE_IMPLEMENTED()` | 任务 1.2：需要根据 `shape`/`strides` 判断是否行主序连续。 |
| `permute(order)`（`tensor.cpp:171`） | ❌ `TO_BE_IMPLEMENTED()` | 任务 1.4：只调整 `shape`/`strides` 顺序，共享 `_storage`。 |
| `view(shape)`（`tensor.cpp:176`） | ❌ `TO_BE_IMPLEMENTED()` | 任务 1.3：合并/拆分维度且不搬数据；若原张量非连续（如转置后的视图）则应报错。 |
| `slice(dim, start, end)`（`tensor.cpp:181`） | ❌ `TO_BE_IMPLEMENTED()` | 任务 1.5：调整对应维度的 `shape` 与 `_offset`，共享 `_storage`。 |
| `load(src)`（`tensor.cpp:186`） | ❌ `TO_BE_IMPLEMENTED()` | 任务 1.1：需要从当前设备 Runtime 拿 API 做 host→device 的 `memcpy`。 |
| `contiguous()` / `reshape(shape)` / `to(device_type, device)`（`tensor.cpp:190/195/200`） | ❌ `TO_BE_IMPLEMENTED()` | 进阶挑战：把非连续张量整理为连续、通用 reshape（必要时拷贝）、跨设备搬运。 |

`view`/`permute`/`slice` 三者应遵循同一原则：**只描述新的形状/步长/偏移，绝不复制底层数据**，因为它们返回的新 `Tensor` 与原张量共享同一个 `_storage`（`std::shared_ptr` 引用计数 +1）。

## 7. C ABI 边界层（`src/llaisys/*.cc`）

这一层是 C++ 世界和外部世界（Python `ctypes` / 未来的其他语言绑定）之间唯一允许出现 `extern "C"` 的地方。核心技巧是一个包装结构体：

```cpp
// src/llaisys/llaisys_tensor.hpp
typedef struct LlaisysTensor {
    llaisys::tensor_t tensor;   // shared_ptr<Tensor>
} LlaisysTensor;
```

`llaisysTensor_t` 在 C API 里只是一个不透明指针（`struct LlaisysTensor *`）。每个导出函数的模式高度一致：

```cpp
llaisysTensor_t tensorView(llaisysTensor_t tensor, size_t *shape, size_t ndim) {
    std::vector<size_t> shape_vec(shape, shape + ndim);      // 1. C 数组 -> std::vector
    return new LlaisysTensor{tensor->tensor->view(shape_vec)}; // 2. 调 C++ 实现 3. 包装成新的不透明指针
}
```

`tensorDestroy` 对应 `delete tensor`——注意这里 `delete` 的只是 `LlaisysTensor` 包装体，真正的张量内存由 `shared_ptr` 引用计数决定何时释放。`src/llaisys/ops.cc` 里的每个 `llaisysXxx` 函数同样只做"解包 `->tensor` → 调 `llaisys::ops::xxx` → （若有返回值）重新包装"，不包含任何业务逻辑；`src/llaisys/runtime.cc` 则直接转发 `llaisysGetRuntimeAPI` / `llaisysSetContextRuntime` 到 `core::context()`。这种"零逻辑边界层"是刻意设计：出问题时可以放心假设 bug 在更下层的 C++ 实现里。

## 8. 算子层设计（`src/ops/`）

每个算子固定包含：`op.hpp`（声明，参数类型都是 `tensor_t`）、`op.cpp`（设备无关的公共层：参数校验 + 按 `deviceType()` 分派）、`cpu/xxx_cpu.hpp`+`.cpp`（CPU 实现，未来还会有 `nvidia/` 子目录）。

### 8.1 校验宏（`src/utils/check.hpp`）

- `CHECK_SAME_DEVICE(a, b, c...)`：所有张量的 `deviceType()`/`deviceId()` 必须一致，否则抛 `EXCEPTION_DEVICE_MISMATCH`。
- `CHECK_SAME_SHAPE(...)` / `CHECK_SAME_DTYPE(...)`：基于同一个 `CHECK_SAME(ERR, FIRST, ...)` 宏模板，用花括号初始化列表遍历比较。
- `ASSERT(condition, message)` / `CHECK_ARGUMENT(condition, message)`：前者用于内部不变式（抛 `runtime_error`），后者用于外部输入校验（抛 `invalid_argument`），都会打印文件名、行号、函数名。
- `TO_BE_IMPLEMENTED()`：统一的"未实现"占位，打印位置信息后 `throw std::runtime_error("Unimplemented function")`——这也是为什么运行到未实现算子时测试会直接崩溃报错，而不是静默返回错误结果。
- `EXCEPTION_UNSUPPORTED_DATATYPE(dtype)` / `EXCEPTION_UNSUPPORTED_DEVICE`：dtype `switch`/设备 `switch` 的 `default` 分支统一使用。

### 8.2 唯一的完整范例：`add`

```cpp
// src/ops/add/op.cpp —— 设备无关公共层
void add(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(), "...");

    if (c->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    }
    llaisys::core::context().setDevice(c->deviceType(), c->deviceId());
    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU: return cpu::add(...);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: TO_BE_IMPLEMENTED(); return;  // 唯一的 stub，只因 NVIDIA 后端未接入
#endif
    default: EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
```

```cpp
// src/ops/add/cpu/add_cpu.cpp —— CPU kernel，按 dtype 分派到模板函数
template <typename T>
void add_(T *c, const T *a, const T *b, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
            c[i] = utils::cast<T>(utils::cast<float>(a[i]) + utils::cast<float>(b[i])); // 升到 float 再算
        } else {
            c[i] = a[i] + b[i];
        }
    }
}
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32: return add_(reinterpret_cast<float*>(c), ...);
    case LLAISYS_DTYPE_BF16: return add_(reinterpret_cast<bf16_t*>(c), ...);
    case LLAISYS_DTYPE_F16: return add_(reinterpret_cast<fp16_t*>(c), ...);
    default: EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
```

`add` 是仓库里唯一端到端跑通的算子，其余算子都应该照此范式实现：**公共层只做校验和分派，具体计算下沉到 `cpu::xxx`，浮点 kernel 内部对 fp16/bf16 都先 `cast<float>` 提升精度再计算，写回时再 `cast<T>` 转回原类型**——这是因为 `fp16_t`/`bf16_t` 只是包了一个 `uint16_t` 的自定义结构体，不支持原生算术运算符，必须借助 `utils::cast<>` 与 IEEE754 转换函数往返。

### 8.3 其余算子当前状态

| 算子 | 头文件签名 | 状态 | 数学契约（详见 README 作业 #2） |
| --- | --- | --- | --- |
| `add` | `add(c, a, b)` | ✅ CPU 完整实现（仅 NVIDIA 分支未实现） | `c = a + b`，逐元素 |
| `argmax` | `argmax(max_idx, max_val, vals)` | ❌ stub | 1D 张量求最大值及下标 |
| `embedding` | `embedding(out, index, weight)` | ❌ stub | 按 `index`(int64) 从 `weight` 表查行 |
| `linear` | `linear(out, in, weight, bias)` | ❌ stub | `Y = X Wᵀ + b`，`bias` 可为空 |
| `rms_norm` | `rms_norm(out, in, weight, eps)` | ❌ stub | 按最后一维做 RMSNorm |
| `rope` | `rope(out, in, pos_ids, theta)` | ❌ stub | 对 Q/K 做旋转位置编码 |
| `self_attention` | `self_attention(attn_val, q, k, v, scale)` | ❌ stub | 因果 softmax 注意力，需自行拼接 KV cache |
| `swiglu` | `swiglu(out, gate, up)` | ❌ stub | `out = up ⊙ silu(gate)` |
| `rearrange` | `rearrange(out, in)` | ❌ stub | 把非连续/不同布局的 `in` 数据整理进连续的 `out` |

课程要求核心浮点算子至少覆盖 **Float32、Float16、BFloat16** 三种 dtype。

## 9. 数据类型系统（`llaisys.h` + `src/utils/`）

`llaisysDataType_t` 是一个覆盖布尔、8~64 位有符号/无符号整数、byte、fp8/fp16/bf16/fp32/fp64、以及 complex16/32/64/128 的枚举（`include/llaisys.h`）。`src/utils/types.hpp` 提供：

- `dsize(dtype)`：返回每种 dtype 的字节数（`switch`，未知类型抛异常）；
- `dtype_to_str(dtype)`：用于报错信息里可读的类型名；
- `fp16_t` / `bf16_t`：都只是 `struct { uint16_t _v; }` 的薄包装，**不重载任何运算符**，因此不能直接 `+`/`*`；
- `utils::cast<TypeTo, TypeFrom>(val)`：一个基于 `if constexpr` 的模板分派器，覆盖了"同类型直通"、"float↔fp16"、"float↔bf16"、"任意类型→fp16/bf16 先转 float 再转"、"fp16/bf16→任意类型先转 float 再 static_cast"等所有组合，底层依赖 `src/utils/types.cpp` 中实现的 `_f16_to_f32`/`_f32_to_f16`/`_bf16_to_f32`/`_f32_to_bf16` 四个真正做位运算的转换函数。

写算子 kernel 时的标准模式就是：**用 `switch(dtype)` 把 `std::byte*` reinterpret 成具体类型指针，模板函数内部对非 float 的窄浮点类型统一 `cast<float>` 升精度计算，最后 `cast<T>` 写回**，`add` 已完整示范了这一模式。

## 10. Python 接口与模型层

Python 包分为两层：`libllaisys` 精确描述 C 类型和函数签名（与 `include/llaisys/*.h` 一一对应）；外层 `RuntimeAPI`、`Tensor`、`Ops` 提供更自然的 Python 调用方式。测试会在 PyTorch 与 LLAISYS 之间搬运相同数据，并比较形状、步长、dtype 和数值结果。

`python/llaisys/models/qwen2.py` 中的 `Qwen2` 类目前只有两个方法的骨架：`__init__` 遍历 safetensors 文件枚举权重名但没有实际加载逻辑，`generate` 直接 `return []`。按作业 #3 要求，模型的前向计算必须用 C/C++ 在 LLAISYS 后端实现，Python 侧只能做胶水代码（不允许借助 PyTorch 等框架实现推理逻辑本身），并且需要实现 KV Cache 否则推理速度会难以接受。

## 11. 构建与运行

前置条件：Xmake、支持 C++17 的编译器、Python 3.9 或更高版本。Python 包声明依赖 PyTorch、Transformers 和 Accelerate。

```bash
# 构建 C++ 共享库
xmake

# 将共享库安装/复制到 Python 包目录（改完 C++ 代码后必须重新执行这一步）
xmake install

# 安装 Python 包及依赖
python -m pip install ./python
```

可以按课程顺序执行测试：

```bash
python test/test_runtime.py --device cpu
python test/test_tensor.py
python test/ops/add.py
python test/ops/argmax.py
python test/ops/embedding.py
python test/ops/linear.py
python test/ops/rms_norm.py
python test/ops/rope.py
python test/ops/self_attention.py
python test/ops/swiglu.py
```

端到端推理测试需要本地模型目录，或者允许脚本从 Hugging Face 下载约 1.5B 参数的模型：

```bash
python test/test_infer.py --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B
```

## 12. 当前实现状态（逐文件核对）

- ✅ 项目分层、公开 C API、CPU Runtime、内存分配（`NaiveAllocator`）、`Context`/`Runtime`/`Storage` 核心框架均已完成且可编译。
- ✅ `add` 算子端到端完整（公共校验 + CPU kernel + fp32/fp16/bf16），是其余算子的实现模板；其 `op.cpp` 中唯一的 `TO_BE_IMPLEMENTED()` 只存在于 `ENABLE_NVIDIA_API` 分支。
- ✅ 张量创建、元信息查询（`shape`/`strides`/`dtype`/`numel`/...）、`debug()`/`info()` 调试打印已经存在。
- ❌ `src/tensor/tensor.cpp` 中 `isContiguous`、`view`、`permute`、`slice`、`load`、`contiguous`、`reshape`、`to` 共 8 个方法仍是 `TO_BE_IMPLEMENTED()`（课程任务 1.1–1.5 及进阶挑战）。
- ❌ 除 `add` 外，`argmax`/`embedding`/`linear`/`rms_norm`/`rope`/`self_attention`/`swiglu`/`rearrange` 的 `op.cpp` 入口全部是 `TO_BE_IMPLEMENTED()`。
- ❌ `python/llaisys/models/qwen2.py` 的权重加载与 `generate` 均为 TODO 占位。
- ⏳ NVIDIA 设备代码：`include`/`src/device/runtime_api.hpp` 中已经预留 `nvidia::getRuntimeAPI()` 声明和 `ENABLE_NVIDIA_API` 宏开关，但 `xmake/nvidia.lua`、`src/device/nvidia/*` 均不存在，`--nv-gpu=y` 目前无法编译通过。

因此，当前仓库更准确的定位是"可编译的教学起点"，而不是已经完成的推理引擎。预置的 `.so` 文件也不能替代从当前源码重新构建和测试，因为它可能与工作区源码状态不同。

## 13. 推荐阅读与实现顺序

1. 先阅读 `include/llaisys.h` 和 `include/llaisys/runtime.h`，理解设备、dtype 和函数表结构。
2. 沿 `python/llaisys/runtime.py` → `src/llaisys/runtime.cc` → `src/core/context/context.cpp` → `src/device/cpu/cpu_runtime_api.cpp` 跟踪一次跨语言调用，理解 `Context`（线程局部单例）→ `Runtime`（单设备资源）→ `LlaisysRuntimeAPI`（函数表）三层关系。
3. 阅读 `src/tensor/tensor.hpp`/`.cpp`，重点理解 `TensorMeta`（dtype/shape/strides）+ `Storage`（共享内存所有权）+ `offset`（视图偏移）三件套，再实现并通过张量测试（`isContiguous`/`view`/`permute`/`slice`/`load`）。
4. 以 `src/ops/add/`（`op.cpp` 校验分派 + `cpu/add_cpu.cpp` 模板 kernel）为模板逐个实现 CPU 算子，注意 fp16/bf16 要借助 `utils::cast<float>` 升精度计算，并用对应 PyTorch 测试验证。
5. 最后实现 Qwen2 权重映射（`python/llaisys/models/qwen2.py` + 对应 C/C++ 后端）、前向传播、KV cache 和 token 生成，再运行端到端推理测试。
6. 若进入作业 #4（CUDA 集成），参考 `src/device/cpu/cpu_runtime_api.cpp` 的写法实现 `src/device/nvidia/`，新增 `xmake/nvidia.lua`，并在每个算子目录下新增 `nvidia/` 子目录接入 CUDA kernel。

## 14. 项目特点与边界

LLAISYS 的优势是规模小、分层清楚、测试直接对齐 PyTorch，适合理解 AI 框架最核心的机制：跨语言 ABI 边界、线程局部的设备上下文管理、张量的 stride/offset/共享存储模型、算子的"校验-分派-kernel"范式，以及窄浮点类型的手工转换。它当前强调正确性和教学可读性，而非生产级性能、完整算子覆盖、自动求导、分布式训练或成熟 GPU 优化（`NaiveAllocator` 不做内存池化就是典型例子）。学习时应把重点放在接口边界、内存布局、设备派发和 Transformer 推理数据流上。
