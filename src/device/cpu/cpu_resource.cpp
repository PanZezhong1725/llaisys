#include "cpu_resource.hpp"

namespace llaisys::device::cpu {
Resource::Resource() : llaisys::device::DeviceResource(LLAISYS_DEVICE_CPU, 0) {}

DeviceResource *getDeviceResource(){
    return new Resource();
}
} // namespace llaisys::device::cpu
