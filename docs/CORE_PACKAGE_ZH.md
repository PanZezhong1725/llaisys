# `src/core/` —— LLAISYS 核心运行时模块

> 本文单独介绍 `src/core/` 这一个软件包，是 `docs/PROJECT_OVERVIEW_ZH.md` 第 5 节内容的展开版。已对照当前仓库源码（`src/core/**`、`include/llaisys/runtime.h`、`xmake.lua`）核对。

## 1. 这个包是做什么的

`src/core` 是整个 LLAISYS 里**唯一负责"设备资源管理"的模块**：它不知道张量（Tensor）长什么样，也不知道任何算子（Op）的计算逻辑，只回答四个问题：

1. 当前线程正在用哪个设备？（`Context`）
2. 这个设备的显存 / 流 / API 函数表在哪？（`Runtime`）
3. 一块分配出来的内存归谁管、什么时候释放？（`Storage`）
4. 内存到底怎么分配、怎么释放？（`MemoryAllocator`）

上层的 `src/tensor`（张量）和 `src/ops`（算子）都构建在这四个概念之上：`Tensor` 内部持有一个 `core::storage_t`（即 `std::shared_ptr<core::Storage>`），算子在派发到具体设备实现前也要通过 `core::context()` 确认/切换当前设备。

## 2. 目录结构

```text
src/core/
├── core.hpp                     # 前向声明 + 全局入口 context()
├── llaisys_core.hpp              # 汇总头文件，外部只需 #include 这一个
├── context/
│   ├── context.hpp / .cpp       # Context：线程局部单例，管理所有设备的 Runtime
├── runtime/
│   ├── runtime.hpp / .cpp       # Runtime：单个设备实例的资源持有者
├── storage/
│   ├── storage.hpp / .cpp       # Storage：一块设备/主机内存的 RAII 包装
└── allocator/
    ├── allocator.hpp            # MemoryAllocator：分配策略的抽象接口
    └── naive_allocator.hpp/.cpp # NaiveAllocator：唯一实现，直接透传给设备 API
```

对应的 xmake 目标是 `llaisys-core`（`xmake.lua:52-66`），静态库，依赖 `llaisys-utils` 和 `llaisys-device`，被 `llaisys-tensor` 依赖：

```text
llaisys-utils  ──┐
llaisys-device ──┼──▶ llaisys-core ──▶ llaisys-tensor ──▶ llaisys-ops ──▶ llaisys（共享库）
```

## 3. 四个核心类的关系

```text
                 thread_local
Context  ───────────────────────▶ 持有每种设备类型 × 每个 device_id 的 Runtime*
   │  setDevice(type, id)             （_runtime_map，惰性创建，当前激活的是 _current_runtime）
   ▼
Runtime  ── 持有一个设备实例的资源：
   ├── const LlaisysRuntimeAPI *_api   （设备无关函数表，来自 src/device）
   ├── llaisysStream_t _stream         （create_stream 得到）
   └── MemoryAllocator *_allocator     （策略对象，当前是 NaiveAllocator）
   │  allocateDeviceStorage(size) / allocateHostStorage(size)
   ▼
Storage  ── RAII 包装一块内存：
   ├── std::byte *_memory
   ├── size_t _size
   ├── Runtime &_runtime     （记住是哪个 Runtime 分配的）
   └── bool _is_host
   析构时自动调用 _runtime.freeStorage(this) 归还内存
```

外部（`src/tensor`）看到的只是 `core::storage_t = std::shared_ptr<Storage>`，多个 `Tensor` 视图（`view`/`permute`/`slice`）可以共享同一个 `Storage`，引用计数归零时才真正释放。

## 4. 逐个组件详解

### 4.1 `Context`（`context/context.hpp` / `.cpp`）

线程局部单例，通过函数内 `static thread_local` 实现：

```cpp
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}
```

- **构造**：遍历所有 `llaisysDeviceType_t`（NVIDIA 等排在前面，`LLAISYS_DEVICE_CPU` 特意放最后作为兜底），对每种设备调用 `get_device_count()` 探测数量，为每个 `device_id` 预留一个 `Runtime*` 槽位。第一个被发现的可用设备会立刻 `new Runtime` 并激活为 `_current_runtime`。由于 CPU 后端 `get_device_count()` 恒为 1，而 NVIDIA 在未实现时返回 0，**默认总是落到 CPU**。
- **`setDevice(device_type, device_id)`**：若与当前激活的 Runtime 不匹配，则 `_deactivate()` 旧的、惰性 `new Runtime(...)`（如果该槽位还没创建过）、再 `_activate()` 新的。
- **`runtime()`**：返回当前激活的 `Runtime&`；如果从未 `setDevice` 过且构造时也没有可用设备，会 `ASSERT` 失败——所以规则是**用之前必须确保有一个激活的 Runtime**（构造函数已经保证至少 CPU 可用）。
- 显式 `delete` 拷贝/移动构造，保证每个线程只有一份状态、且不会被意外复制。

### 4.2 `Runtime`（`runtime/runtime.hpp` / `.cpp`）

代表"一个设备实例"的资源持有者，构造函数私有，只能由 `Context`（`friend`）创建：

```cpp
Runtime(llaisysDeviceType_t device_type, int device_id)
    : _api(llaisys::device::getRuntimeAPI(device_type)),
      _stream(_api->create_stream()),
      _allocator(new allocators::NaiveAllocator(_api)) {}
```

对外方法：

| 方法 | 作用 |
| --- | --- |
| `deviceType()` / `deviceId()` / `isActive()` | 查询自身状态 |
| `api()` | 拿到底层 `LlaisysRuntimeAPI` 函数表指针，供算子层直接调用（如 `memcpy_sync`） |
| `allocateDeviceStorage(size)` | 走 `_allocator->allocate(size)`，包成 `Storage`，`is_host=false` |
| `allocateHostStorage(size)` | 走 `_api->malloc_host(size)`，包成 `Storage`，`is_host=true` |
| `freeStorage(Storage*)` | 根据 `storage->isHost()` 决定走 `_api->free_host()` 还是 `_allocator->release()`，由 `Storage` 析构时自动触发 |
| `stream()` / `synchronize()` | 暴露当前流，或阻塞等待其上所有操作完成 |

析构时 `delete _allocator` 并 `_api->destroy_stream(_stream)`；如果析构时 `_is_active` 仍是 `false`，会打印一条警告（当前实现里只是 `std::cerr`，不算致命错误）。

### 4.3 `Storage`（`storage/storage.hpp` / `.cpp`）

一块内存的 RAII 包装，**构造函数私有，只有 `Runtime` 是 `friend`**——这保证了"内存块的生命周期必须由分配它的 `Runtime` 来管理"这一不变式，外部代码不可能绕过 `Runtime` 直接构造出一个 `Storage`。

```cpp
class Storage {
    std::byte *_memory; size_t _size; Runtime &_runtime; bool _is_host;
    ~Storage() { _runtime.freeStorage(this); }   // 析构自动归还内存
public:
    std::byte *memory() const;
    size_t size() const;
    llaisysDeviceType_t deviceType() const;  // is_host 时恒为 CPU
    int deviceId() const;                    // is_host 时恒为 0
    bool isHost() const;
};
```

`Tensor` 通过 `core::storage_t`（`std::shared_ptr<Storage>`）持有它；`view`/`permute`/`slice` 产生的多个 `Tensor` 视图可以共享同一个 `Storage`，最后一个 `shared_ptr` 析构时才真正释放内存——这是整个项目里内存自动管理的根基。

### 4.4 `MemoryAllocator` / `NaiveAllocator`（`allocator/`）

`MemoryAllocator` 是分配策略的抽象接口，只有两个纯虚函数：

```cpp
class MemoryAllocator {
protected:
    const LlaisysRuntimeAPI *_api;
public:
    virtual std::byte *allocate(size_t size) = 0;
    virtual void release(std::byte *memory) = 0;
};
```

当前唯一实现 `NaiveAllocator` 直接透传给 `_api->malloc_device()` / `_api->free_device()`，**不做池化、不做复用**——这是刻意简化的教学版本。如果要做显存池、按大小分级复用等性能优化（作业范围之外的扩展方向），`MemoryAllocator` 这个接口就是天然的切入点：只需新写一个子类，在 `Runtime` 构造函数里换成新的实现即可，`Runtime`/`Storage`/`Context` 都不需要改动。

### 4.5 `LlaisysRuntimeAPI`：设备无关的函数表（跨到 `src/device`）

`core` 包本身不实现任何设备相关的系统调用，而是通过 `include/llaisys/runtime.h` 定义的一张 12 个函数指针组成的 C 结构体来间接调用：

```c
struct LlaisysRuntimeAPI {
    get_device_count_api get_device_count;   set_device_api set_device;
    device_synchronize_api device_synchronize;
    create_stream_api create_stream;         destroy_stream_api destroy_stream;
    stream_synchronize_api stream_synchronize;
    malloc_device_api malloc_device;         free_device_api free_device;
    malloc_host_api malloc_host;             free_host_api free_host;
    memcpy_sync_api memcpy_sync;             memcpy_async_api memcpy_async;
};
```

`Runtime` 构造时通过 `llaisys::device::getRuntimeAPI(device_type)`（`src/device/runtime_api.cpp`）拿到这张表：

- `LLAISYS_DEVICE_CPU` → CPU 实现（`src/device/cpu/`），全部基于 `std::malloc/std::free/std::memcpy`，"设备内存"和"主机内存"其实是同一块内存；
- `LLAISYS_DEVICE_NVIDIA` → 若编译时定义了 `ENABLE_NVIDIA_API` 则调用 NVIDIA 实现，否则退回 `getUnsupportedRuntimeAPI()`（所有函数都 `throw std::runtime_error`，用于给出明确报错而非链接失败）。

这种"胖接口 + 函数表分发"的设计，使得 `core` 里的 `Context`/`Runtime`/`Storage`/`Allocator` **完全不需要关心具体是哪种设备**，新增一种设备后端只需要在 `src/device` 下补一张函数表，`core` 包不用改一行代码。

## 5. 典型调用时序

以"创建一个 CPU 张量"为例（发生在 `src/tensor/tensor.cpp` 的 `Tensor::create` 里），串起 `core` 包的四个类：

```text
Tensor::create(shape, dtype, LLAISYS_DEVICE_CPU, 0)
        │
        ▼
core::context()                       // 拿到当前线程的 Context（首次调用触发构造，默认已激活 CPU Runtime）
        │
        ▼
context().setDevice(LLAISYS_DEVICE_CPU, 0)   // 若已经是 CPU，直接跳过
        │
        ▼
context().runtime()                    // 拿到当前激活的 Runtime&
        │
        ▼
runtime.allocateDeviceStorage(bytes)   // Runtime -> NaiveAllocator::allocate -> api->malloc_device
        │
        ▼
new Storage(ptr, size, runtime, /*is_host=*/false)   // Runtime 是 friend，可以调用私有构造函数
        │
        ▼
返回 shared_ptr<Storage> 给 Tensor 持有
```

当 `shared_ptr<Storage>` 引用计数归零时，`~Storage()` 自动调用 `runtime.freeStorage(this)`，再根据 `is_host` 决定走分配器释放还是 `free_host`——**整条链路上没有任何一处需要手动 `delete` 内存**。

## 6. 使用 `core` 包时需要记住的几条规则

- **必须先有激活的 Runtime 才能 `context().runtime()`**：`Context` 构造时已保证至少 CPU 可用，正常使用不需要手动处理，但如果要切到非默认设备，记得先 `setDevice()`。
- **`Storage`/`Runtime`/`Context` 都不可拷贝、不可移动**，全部通过指针/引用/`shared_ptr` 传递，这是为了保证"一块内存只有一个归属者"的不变式。
- **`Storage` 的构造函数是私有的**，只能通过 `Runtime::allocateDeviceStorage` / `allocateHostStorage` 获得，不要试图绕开 `Runtime` 直接构造。
- **`MemoryAllocator` 是唯一为将来扩展预留的接口**：如果要实现显存池等优化，只需要新增一个 `MemoryAllocator` 子类并在 `Runtime` 构造函数里替换 `NaiveAllocator`。
- **`core` 包完全不知道 `Tensor` 的存在**：它只提供"内存在哪、怎么分配、什么时候释放"这三件事，任何张量形状/步长/dtype 相关的逻辑都在 `src/tensor` 层，不应该也不需要下沉到这里。
