#include "runtime.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/naive_allocator.hpp"

namespace llaisys::core {
Runtime::Runtime(llaisysDeviceType_t device_type, int device_id)
    : _device_type(device_type), _device_id(device_id), _is_active(false) {
    _api = llaisys::device::getRuntimeAPI(_device_type);
    _stream = _api->create_stream();
    _allocator = new allocators::NaiveAllocator(_api);
}

Runtime::~Runtime() {
    if (!_is_active) {
        std::cerr << "Mallicious destruction of inactive runtime." << std::endl;
    }
    delete _allocator;
    _allocator = nullptr;
    _api->destroy_stream(_stream);
    _api = nullptr;
}

void Runtime::_activate() {
    _api->set_device(_device_id);
    _is_active = true;
}

/*
 * _deactivate() 没有调用底层“关闭设备”函数，只是更新逻辑状态
 * 因为设备切换通常是：把另一个设备设置为 current，而不是“关闭原设备”
 */
void Runtime::_deactivate() {
    _is_active = false;
}

bool Runtime::isActive() const {
    return _is_active;
}

llaisysDeviceType_t Runtime::deviceType() const {
    return _device_type;
}

int Runtime::deviceId() const {
    return _device_id;
}

const LlaisysRuntimeAPI *Runtime::api() const {
    return _api;
}

//  Device Storage：给算子在设备上直接计算
storage_t Runtime::allocateDeviceStorage(size_t size) {
    // _allocator->allocate 表示通过当前 Runtime 的设备内存分配器分配内存
    return std::shared_ptr<Storage>(new Storage(_allocator->allocate(size), size, *this, false));
}

// Host Storage：给 CPU 访问、加载数据和设备传输
storage_t Runtime::allocateHostStorage(size_t size) {
    return std::shared_ptr<Storage>(new Storage((std::byte *)_api->malloc_host(size), size, *this, true));
}

void Runtime::freeStorage(Storage *storage) {
    if (storage->isHost()) {
        _api->free_host(storage->memory());
    } else {
        _allocator->release(storage->memory());
    }
}

llaisysStream_t Runtime::stream() const {
    return _stream;
}

void Runtime::synchronize() const {
    _api->stream_synchronize(_stream);
}

} // namespace llaisys::core
