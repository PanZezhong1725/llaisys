#pragma once
#include "llaisys.h"

#include "../utils.hpp"

namespace llaisys::device {
class DeviceResource {
private:
    llaisysDeviceType_t _device_type;
    int _device_id;

public:
    DeviceResource(llaisysDeviceType_t device_type, int device_id)
        : _device_type(device_type),
          _device_id(device_id) {
    }
    virtual ~DeviceResource() = default;

    llaisysDeviceType_t getDeviceType() const { return _device_type; }
    int getDeviceId() const { return _device_id; };
};

DeviceResource *getDeviceResource(llaisysDeviceType_t device_type, int device_id);
namespace cpu { DeviceResource *getDeviceResource(); }
#ifdef ENABLE_NVIDIA_API
namespace nvidia { DeviceResource *getDeviceResource(int device_id); }
#endif
#ifdef ENABLE_ILUVATAR_API
namespace iluvatar { DeviceResource *getDeviceResource(int device_id); }
#endif


} // namespace llaisys::device
