#include "runtime.hpp"

#include "../allocator/naive_allocator.hpp"

namespace llaisys::core {

Runtime::Runtime(llaisysDeviceType_t device_type, int device_id)
    : _device_type(device_type),
      _device_id(device_id),
      _api(llaisys::device::getRuntimeAPI(device_type)),
      _allocator(nullptr),
      _is_active(false),
      _stream(nullptr) {
    _api->set_device(_device_id);
    _stream = _api->create_stream();
    _allocator = new allocators::NaiveAllocator(_api);
}

Runtime::~Runtime() {
    _api->set_device(_device_id);
    delete _allocator;
    _allocator = nullptr;
    _api->destroy_stream(_stream);
    _stream = nullptr;
}

void Runtime::_activate() {
    _api->set_device(_device_id);
    _is_active = true;
}

void Runtime::_deactivate() {
    _is_active = false;
}

bool Runtime::isActive() const { return _is_active; }
llaisysDeviceType_t Runtime::deviceType() const { return _device_type; }
int Runtime::deviceId() const { return _device_id; }
const LlaisysRuntimeAPI *Runtime::api() const { return _api; }

storage_t Runtime::allocateDeviceStorage(size_t size) {
    auto *memory = size == 0 ? nullptr : _allocator->allocate(size);
    return storage_t(new Storage(memory, size, *this, false));
}

storage_t Runtime::allocateHostStorage(size_t size) {
    auto *memory = size == 0 ? nullptr : static_cast<std::byte *>(_api->malloc_host(size));
    return storage_t(new Storage(memory, size, *this, true));
}

void Runtime::freeStorage(Storage *storage) {
    _api->set_device(_device_id);
    if (storage->memory() == nullptr) {
        return;
    }
    if (storage->isHost()) {
        _api->free_host(storage->memory());
    } else {
        _allocator->release(storage->memory());
    }
}

llaisysStream_t Runtime::stream() const { return _stream; }

void Runtime::synchronize() const {
    _api->stream_synchronize(_stream);
}

} // namespace llaisys::core
