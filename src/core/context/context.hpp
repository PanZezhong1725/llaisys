#pragma once

#include "llaisys.h"

#include "../core.hpp"

#include "../runtime/runtime.hpp"

#include <unordered_map>
#include <vector>

namespace llaisys::core {
/*
 * Context 是每个线程的设备状态管理器，负责保存各设备 Runtime，并选出当前 Runtime
 */
class Context {
private:
    // 每一种 Device 对应一个 Runtime 组别
    std::unordered_map<llaisysDeviceType_t, std::vector<Runtime *>> _runtime_map;
    // 当前的 Runtime，由 (Device_type, Device_id) 指定
    Runtime *_current_runtime;
    Context();

public:
    ~Context();

    // Prevent copy
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    // Prevent move
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;

    void setDevice(llaisysDeviceType_t device_type, int device_id);
    Runtime &runtime();

    friend Context &context();
};
} // namespace llaisys::core
