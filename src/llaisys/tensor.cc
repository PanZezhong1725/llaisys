#include "llaisys_tensor.hpp"

#include <exception>
#include <iostream>
#include <vector>

namespace {

void reportTensorError(const char *api, const std::exception &error) noexcept {
    std::cerr << "[ERROR] " << api << " failed: " << error.what() << std::endl;
}

void reportUnknownTensorError(const char *api) noexcept {
    std::cerr << "[ERROR] " << api << " failed with an unknown exception." << std::endl;
}

} // namespace

__C {
    llaisysTensor_t tensorCreate(
        size_t * shape,
        size_t ndim,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id) {
        try {
            if (ndim != 0 && shape == nullptr) {
                return nullptr;
            }
            std::vector<size_t> shape_vec;
            if (ndim != 0) {
                shape_vec.assign(shape, shape + ndim);
            }
            return new LlaisysTensor{
                llaisys::Tensor::create(shape_vec, dtype, device_type, device_id)};
        } catch (const std::exception &error) {
            reportTensorError("tensorCreate", error);
            return nullptr;
        } catch (...) {
            reportUnknownTensorError("tensorCreate");
            return nullptr;
        }
    }

    void tensorDestroy(
        llaisysTensor_t tensor) {
        delete tensor;
    }

    void *tensorGetData(
        llaisysTensor_t tensor) {
        return tensor->tensor->data();
    }

    size_t tensorGetNdim(
        llaisysTensor_t tensor) {
        return tensor->tensor->ndim();
    }

    void tensorGetShape(
        llaisysTensor_t tensor,
        size_t * shape) {
        std::copy(tensor->tensor->shape().begin(), tensor->tensor->shape().end(), shape);
    }

    void tensorGetStrides(
        llaisysTensor_t tensor,
        ptrdiff_t * strides) {
        std::copy(tensor->tensor->strides().begin(), tensor->tensor->strides().end(), strides);
    }

    llaisysDataType_t tensorGetDataType(
        llaisysTensor_t tensor) {
        return tensor->tensor->dtype();
    }

    llaisysDeviceType_t tensorGetDeviceType(
        llaisysTensor_t tensor) {
        return tensor->tensor->deviceType();
    }

    int tensorGetDeviceId(
        llaisysTensor_t tensor) {
        return tensor->tensor->deviceId();
    }

    void tensorDebug(
        llaisysTensor_t tensor) {
        tensor->tensor->debug();
    }

    uint8_t tensorIsContiguous(
        llaisysTensor_t tensor) {
        return uint8_t(tensor->tensor->isContiguous());
    }

    void tensorLoad(
        llaisysTensor_t tensor,
        const void *data) {
        tensor->tensor->load(data);
    }

    llaisysTensor_t tensorView(
        llaisysTensor_t tensor,
        size_t * shape,
        size_t ndim) {
        try {
            if (tensor == nullptr || (ndim != 0 && shape == nullptr)) {
                return nullptr;
            }
            std::vector<size_t> shape_vec;
            if (ndim != 0) {
                shape_vec.assign(shape, shape + ndim);
            }
            return new LlaisysTensor{tensor->tensor->view(shape_vec)};
        } catch (const std::exception &error) {
            reportTensorError("tensorView", error);
            return nullptr;
        } catch (...) {
            reportUnknownTensorError("tensorView");
            return nullptr;
        }
    }

    llaisysTensor_t tensorPermute(
        llaisysTensor_t tensor,
        size_t * order) {
        try {
            if (tensor == nullptr
                || (tensor->tensor->ndim() != 0 && order == nullptr)) {
                return nullptr;
            }
            std::vector<size_t> order_vec;
            if (tensor->tensor->ndim() != 0) {
                order_vec.assign(order, order + tensor->tensor->ndim());
            }
            return new LlaisysTensor{tensor->tensor->permute(order_vec)};
        } catch (const std::exception &error) {
            reportTensorError("tensorPermute", error);
            return nullptr;
        } catch (...) {
            reportUnknownTensorError("tensorPermute");
            return nullptr;
        }
    }

    llaisysTensor_t tensorSlice(
        llaisysTensor_t tensor,
        size_t dim,
        size_t start,
        size_t end) {
        try {
            if (tensor == nullptr) {
                return nullptr;
            }
            return new LlaisysTensor{tensor->tensor->slice(dim, start, end)};
        } catch (const std::exception &error) {
            reportTensorError("tensorSlice", error);
            return nullptr;
        } catch (...) {
            reportUnknownTensorError("tensorSlice");
            return nullptr;
        }
    }
}
