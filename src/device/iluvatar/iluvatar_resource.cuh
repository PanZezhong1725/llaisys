#pragma once
#include <cublas_v2.h>
#include "../device_resource.hpp"

namespace llaisys::device::iluvatar {
class Resource : public llaisys::device::DeviceResource {
private:
    cublasHandle_t  _cublas_handle;
public:
    Resource(int device_id);
    ~Resource();


    cublasHandle_t cublasHandle() const {
        return _cublas_handle;
    }
};
} // namespace llaisys::device::iluvatar
