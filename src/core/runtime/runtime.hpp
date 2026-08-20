#pragma once
#include "../core.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/allocator.hpp"

namespace llaisys::core {
class Runtime {
private:
    llaisysDeviceType_t _device_type; // 设备类型，如 NV，MX
    int _device_id;                   // 设备编号，如 0，1，2
    const LlaisysRuntimeAPI *_api;    // 本质上是一个函数指针表，类似虚函数接口
    MemoryAllocator *_allocator;      // 当前 NaiveAllocator 非常简单：api->malloc_device(size) api->frees(size)
                                      // 设备内存分配不是直接在 Tensor 中完成
                                      // Tensor
                                      //    → Runtime
                                      //      → Allocator
                                      //        → Runtime API
                                      //          → malloc_device
    bool _is_active;
    void _activate();
    void _deactivate();
    llaisysStream_t _stream;
    Runtime(llaisysDeviceType_t device_type, int device_id);

public:
    // 构造函数是 private 的，意味着外部无法直接 new Runtime(...)
    // 但将 Context 类设为了友元类，这说明 Runtime 实例只能由 Context 来统一创建和管理
    friend class Context;

    ~Runtime();

    // Prevent copying
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    // Prevent moving
    Runtime(Runtime &&) = delete;
    Runtime &operator=(Runtime &&) = delete;

    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    bool isActive() const;

    const LlaisysRuntimeAPI *api() const;

    storage_t allocateDeviceStorage(size_t size);
    ;
    storage_t allocateHostStorage(size_t size);
    void freeStorage(Storage *storage);

    llaisysStream_t stream() const;
    void synchronize() const;
};
} // namespace llaisys::core
