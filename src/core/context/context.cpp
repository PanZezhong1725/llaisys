#include "context.hpp"

#include "../../utils.hpp"

#include <utility>

namespace llaisys::core {

Context::Context() : _current_runtime(nullptr) {
    // Prefer an accelerator when the library was built with one, while always
    // retaining a CPU runtime as the portable fallback.
    std::vector<llaisysDeviceType_t> probe_order;
    for (int value = 1; value < LLAISYS_DEVICE_TYPE_COUNT; ++value) {
        probe_order.push_back(static_cast<llaisysDeviceType_t>(value));
    }
    probe_order.push_back(LLAISYS_DEVICE_CPU);

    for (const auto type : probe_order) {
        const auto *api = llaisysGetRuntimeAPI(type);
        const int count = api->get_device_count();
        CHECK_ARGUMENT(count >= 0, "device count cannot be negative");
        _runtime_map.emplace(type, std::vector<Runtime *>(static_cast<size_t>(count), nullptr));

        if (_current_runtime == nullptr && count != 0) {
            auto *first = new Runtime(type, 0);
            first->_activate();
            _runtime_map.at(type).front() = first;
            _current_runtime = first;
        }
    }
}

Context::~Context() {
    for (auto &[type, slots] : _runtime_map) {
        (void)type;
        for (auto *runtime : slots) {
            if (runtime == nullptr) {
                continue;
            }
            runtime->_activate();
            delete runtime;
        }
    }
    _current_runtime = nullptr;
    _runtime_map.clear();
}

void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
    const auto found = _runtime_map.find(device_type);
    CHECK_ARGUMENT(found != _runtime_map.end(), "unknown device type");
    auto &slots = found->second;
    CHECK_ARGUMENT(
        device_id >= 0 && static_cast<size_t>(device_id) < slots.size(),
        "invalid device id");

    Runtime *selected = slots[static_cast<size_t>(device_id)];
    if (selected == nullptr) {
        selected = new Runtime(device_type, device_id);
        slots[static_cast<size_t>(device_id)] = selected;
    }

    if (_current_runtime != nullptr && _current_runtime != selected) {
        _current_runtime->_deactivate();
    }
    // Re-activate even when the cached runtime is selected. The public runtime
    // API can change the process' current CUDA device independently.
    selected->_activate();
    _current_runtime = selected;
}

Runtime &Context::runtime() {
    ASSERT(_current_runtime != nullptr, "No runtime is active.");
    return *_current_runtime;
}

Context &context() {
    thread_local Context instance;
    return instance;
}

} // namespace llaisys::core
