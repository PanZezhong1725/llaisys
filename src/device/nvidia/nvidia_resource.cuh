#pragma once

#include "../device_resource.hpp"

#include <cublas_v2.h>

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();
};

// Get the thread-local cuBLAS handle for the current device.
// The handle is created lazily on first use and destroyed when the thread exits.
cublasHandle_t getCublasHandle();
} // namespace llaisys::device::nvidia
