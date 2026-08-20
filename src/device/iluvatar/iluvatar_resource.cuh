#pragma once

#include "../device_resource.hpp"

#include <cublas_v2.h>

namespace llaisys::device::iluvatar {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();
};

cublasHandle_t getCublasHandle();

} // namespace llaisys::device::iluvatar
