#pragma once

#include "../device_resource.hpp"

#include <cublas_v2.h>

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
private:
    cublasHandle_t _cublas;

public:
    Resource(int device_id);
    ~Resource();

    cublasHandle_t cublas(llaisysStream_t stream);
};

Resource &getResource(int device_id);
} // namespace llaisys::device::nvidia
