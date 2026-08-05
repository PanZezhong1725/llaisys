#include "device_resource.hpp"

namespace llaisys::device {

DeviceResource * getDeviceResource(llaisysDeviceType_t device_type, int device_id){
    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return cpu::getDeviceResource();
    case LLAISYS_DEVICE_NVIDIA:
#ifdef ENABLE_NVIDIA_API
        return nvidia::getDeviceResource(device_id);
#else
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
#endif
    case LLAISYS_DEVICE_ILUVATAR:
#ifdef ENABLE_ILUVATAR_API
        return iluvatar::getDeviceResource(device_id);
#else
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
    }
}
}
