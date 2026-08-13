#include "tensor.hpp"
#include "../utils.hpp"
#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() { return _storage->memory() + _offset; }
const std::byte *Tensor::data() const { return _storage->memory() + _offset; }
size_t Tensor::ndim() const { return _meta.shape.size(); }
const std::vector<size_t> &Tensor::shape() const { return _meta.shape; }
const std::vector<ptrdiff_t> &Tensor::strides() const { return _meta.strides; }
llaisysDataType_t Tensor::dtype() const { return _meta.dtype; }
llaisysDeviceType_t Tensor::deviceType() const { return _storage->deviceType(); }
int Tensor::deviceId() const { return _storage->deviceId(); }
size_t Tensor::numel() const { return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>()); }
size_t Tensor::elementSize() const { return utils::dsize(_meta.dtype); }

void Tensor::load(const void *src_) {
    size_t total_bytes = this->numel() * this->elementSize();
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(this->data(), src_, total_bytes);
    } else {
        core::context().setDevice(this->deviceType(), this->deviceId());
        core::context().runtime().api()->memcpy_sync(
            this->data(), src_, total_bytes, LLAISYS_MEMCPY_H2D);
    }
}

bool Tensor::isContiguous() const {
    size_t ndim = _meta.shape.size();
    if (ndim <= 1) return true;
    ptrdiff_t expected_stride = 1;
    for (int i = (int)ndim - 1; i >= 0; --i) {
        if (_meta.strides[i] != expected_stride) return false;
        expected_stride *= (ptrdiff_t)_meta.shape[i];
    }
    return true;
}

// Simple info for debug
std::string Tensor::info() const {
    std::stringstream ss;
    ss << "Tensor: shape[";
    for (auto s : _meta.shape) ss << s << " ";
    ss << "] strides[";
    for (auto s : _meta.strides) ss << s << " ";
    ss << "] dtype=" << _meta.dtype;
    return ss.str();
}

void Tensor::debug() const {
    std::cout << this->info() << std::endl;
    // Simplified: just print first few floats if on CPU
    if (this->deviceType() == LLAISYS_DEVICE_CPU && this->numel() > 0 && _meta.dtype == 6) {
        auto *fd = (const float*)this->data();
        size_t n = std::min(this->numel(), (size_t)20);
        for (size_t i = 0; i < n; i++) std::cout << fd[i] << " ";
        std::cout << std::endl;
    }
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim = this->ndim();
    std::vector<size_t> new_shape(ndim);
    std::vector<ptrdiff_t> new_strides(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    size_t new_numel = 1;
    for (size_t s : shape) new_numel *= s;
    if (new_numel != this->numel())
        throw std::invalid_argument("view: element count mismatch");
    size_t ndim = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim);
    ptrdiff_t stride = 1;
    for (int i = (int)ndim - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= (ptrdiff_t)shape[i];
    }
    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    TensorMeta new_meta = _meta;
    new_meta.shape[dim] = end - start;
    size_t byte_offset = _offset + start * _meta.strides[dim] * this->elementSize();
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, byte_offset));
}

tensor_t Tensor::contiguous() const {
    if (this->isContiguous())
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    auto new_tensor = Tensor::create(this->shape(), this->dtype(), this->deviceType(), this->deviceId());
    size_t ndim = this->ndim(), elem_size = this->elementSize(), total = this->numel();
    const auto &shape = this->shape();
    const auto &strides = this->strides();
    const std::byte *src_base = this->data();
    std::byte *dst_base = new_tensor->data();
    std::vector<size_t> indices(ndim, 0);
    for (size_t i = 0; i < total; ++i) {
        size_t src_elem_offset = 0;
        for (size_t d = 0; d < ndim; ++d) src_elem_offset += indices[d] * strides[d];
        std::memcpy(dst_base + i * elem_size, src_base + src_elem_offset * elem_size, elem_size);
        for (int d = (int)ndim - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < shape[d]) break;
            indices[d] = 0;
        }
    }
    return new_tensor;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    return this->contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    if (this->deviceType() == device_type && this->deviceId() == device)
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    auto new_tensor = Tensor::create(this->shape(), this->dtype(), device_type, device);
    size_t sz = this->numel() * this->elementSize();
    if (this->deviceType() == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU)
        std::memcpy(new_tensor->data(), this->data(), sz);
    else if (this->deviceType() == LLAISYS_DEVICE_NVIDIA || device_type == LLAISYS_DEVICE_NVIDIA) {
        core::context().setDevice(device_type, device);
        auto kind = (this->deviceType() == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2D :
                    (device_type == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_D2H : LLAISYS_MEMCPY_D2D;
        core::context().runtime().api()->memcpy_sync(new_tensor->data(), this->data(), sz, kind);
    }
    return new_tensor;
}

} // namespace llaisys