#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

namespace llaisys {

namespace {

size_t checkedMultiply(size_t lhs, size_t rhs, const char *message) {
    CHECK_ARGUMENT(
        rhs == 0 || lhs <= std::numeric_limits<size_t>::max() / rhs,
        message);
    return lhs * rhs;
}

size_t checkedAdd(size_t lhs, size_t rhs, const char *message) {
    CHECK_ARGUMENT(
        lhs <= std::numeric_limits<size_t>::max() - rhs,
        message);
    return lhs + rhs;
}

size_t checkedNumel(const std::vector<size_t> &shape) {
    // A zero-sized dimension makes the tensor empty regardless of the other
    // dimensions. Checking this first also avoids rejecting a valid empty
    // tensor merely because dimensions to the left would overflow a product.
    for (size_t dim_size : shape) {
        if (dim_size == 0) {
            return 0;
        }
    }

    size_t result = 1;
    for (size_t dim_size : shape) {
        result = checkedMultiply(result, dim_size, "tensor element count overflows size_t");
    }
    return result;
}

std::vector<ptrdiff_t> makeContiguousStrides(const std::vector<size_t> &shape) {
    std::vector<ptrdiff_t> strides(shape.size());
    size_t stride = 1;

    for (size_t i = shape.size(); i > 0; --i) {
        const size_t dim = i - 1;
        CHECK_ARGUMENT(
            stride <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()),
            "tensor stride overflows ptrdiff_t");
        strides[dim] = static_cast<ptrdiff_t>(stride);
        stride = checkedMultiply(stride, shape[dim], "tensor stride overflows size_t");
    }
    return strides;
}

bool computeViewStrides(
    const std::vector<size_t> &old_shape,
    const std::vector<ptrdiff_t> &old_strides,
    const std::vector<size_t> &new_shape,
    std::vector<ptrdiff_t> &new_strides) {
    CHECK_ARGUMENT(
        old_shape.size() == old_strides.size(),
        "tensor shape and strides must have the same rank");

    const size_t old_numel = checkedNumel(old_shape);
    if (checkedNumel(new_shape) != old_numel) {
        return false;
    }

    if (old_numel == 0) {
        // Preserve metadata for a no-op view. For a different empty shape,
        // canonical strides are unambiguous because no element is addressable.
        new_strides = new_shape == old_shape
                        ? old_strides
                        : makeContiguousStrides(new_shape);
        return true;
    }

    if (old_shape.empty()) {
        // A scalar has one element and can be split into any all-ones shape.
        new_strides = makeContiguousStrides(new_shape);
        return true;
    }

    CHECK_ARGUMENT(
        old_shape.size() <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())
            && new_shape.size() <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()),
        "tensor rank is too large");

    new_strides.assign(new_shape.size(), 0);
    ptrdiff_t view_dim = static_cast<ptrdiff_t>(new_shape.size()) - 1;
    ptrdiff_t chunk_base_stride = old_strides.back();
    if (chunk_base_stride < 0) {
        return false;
    }

    size_t tensor_chunk_numel = 1;
    size_t view_chunk_numel = 1;

    for (ptrdiff_t tensor_dim = static_cast<ptrdiff_t>(old_shape.size()) - 1;
         tensor_dim >= 0;
         --tensor_dim) {
        const size_t dim = static_cast<size_t>(tensor_dim);
        tensor_chunk_numel = checkedMultiply(
            tensor_chunk_numel,
            old_shape[dim],
            "view chunk element count overflows size_t");

        bool chunk_boundary = tensor_dim == 0;
        if (!chunk_boundary && old_shape[dim - 1] != 1) {
            if (old_strides[dim - 1] < 0) {
                return false;
            }
            const size_t expected_stride = checkedMultiply(
                tensor_chunk_numel,
                static_cast<size_t>(chunk_base_stride),
                "view stride overflows size_t");
            if (expected_stride > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())
                || old_strides[dim - 1] != static_cast<ptrdiff_t>(expected_stride)) {
                chunk_boundary = true;
            }
        }

        if (!chunk_boundary) {
            continue;
        }

        while (view_dim >= 0
               && (view_chunk_numel < tensor_chunk_numel
                   || new_shape[static_cast<size_t>(view_dim)] == 1)) {
            const size_t new_dim = static_cast<size_t>(view_dim);
            const size_t stride = checkedMultiply(
                view_chunk_numel,
                static_cast<size_t>(chunk_base_stride),
                "view stride overflows size_t");
            CHECK_ARGUMENT(
                stride <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()),
                "view stride overflows ptrdiff_t");
            new_strides[new_dim] = static_cast<ptrdiff_t>(stride);
            view_chunk_numel = checkedMultiply(
                view_chunk_numel,
                new_shape[new_dim],
                "view chunk element count overflows size_t");
            --view_dim;
        }

        if (view_chunk_numel != tensor_chunk_numel) {
            return false;
        }

        if (tensor_dim > 0) {
            chunk_base_stride = old_strides[dim - 1];
            if (chunk_base_stride < 0) {
                return false;
            }
            tensor_chunk_numel = 1;
            view_chunk_numel = 1;
        }
    }

    return view_dim == -1;
}

void copyLogicalToContiguous(const Tensor &source, Tensor &destination) {
    CHECK_ARGUMENT(source.shape() == destination.shape(), "tensor copy shape mismatch");
    CHECK_ARGUMENT(source.dtype() == destination.dtype(), "tensor copy dtype mismatch");
    CHECK_ARGUMENT(destination.isContiguous(), "tensor copy destination must be contiguous");
    CHECK_ARGUMENT(
        source.deviceType() == destination.deviceType()
            && source.deviceId() == destination.deviceId(),
        "logical tensor copy requires tensors on the same device");

    const size_t numel = source.numel();
    if (numel == 0) {
        return;
    }

    for (ptrdiff_t stride : source.strides()) {
        CHECK_ARGUMENT(stride >= 0, "negative tensor strides are not supported");
    }

    // Copy the largest contiguous trailing chunk at once. This keeps slices
    // efficient while still handling arbitrary permutations correctly.
    size_t chunk_elements = 1;
    size_t chunk_start_dim = source.ndim();
    for (size_t i = source.ndim(); i > 0; --i) {
        const size_t dim = i - 1;
        if (source.shape()[dim] == 1) {
            chunk_start_dim = dim;
            continue;
        }
        if (chunk_elements > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())
            || source.strides()[dim] != static_cast<ptrdiff_t>(chunk_elements)) {
            break;
        }
        chunk_elements = checkedMultiply(
            chunk_elements,
            source.shape()[dim],
            "tensor copy chunk size overflows size_t");
        chunk_start_dim = dim;
    }

    const size_t chunk_count = numel / chunk_elements;
    const size_t chunk_bytes = checkedMultiply(
        chunk_elements,
        source.elementSize(),
        "tensor copy byte count overflows size_t");
    CHECK_ARGUMENT(source.data() != nullptr && destination.data() != nullptr, "tensor data pointer is null");

    const bool on_cpu = source.deviceType() == LLAISYS_DEVICE_CPU;
    core::Runtime *runtime = nullptr;
    if (!on_cpu) {
        core::context().setDevice(source.deviceType(), source.deviceId());
        runtime = &core::context().runtime();
    }

    size_t queued_copies = 0;
    for (size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        size_t remaining = chunk_index;
        size_t source_element_offset = 0;
        for (size_t i = chunk_start_dim; i > 0; --i) {
            const size_t dim = i - 1;
            const size_t index = remaining % source.shape()[dim];
            remaining /= source.shape()[dim];
            const size_t contribution = checkedMultiply(
                index,
                static_cast<size_t>(source.strides()[dim]),
                "tensor copy source offset overflows size_t");
            source_element_offset = checkedAdd(
                source_element_offset,
                contribution,
                "tensor copy source offset overflows size_t");
        }

        const size_t source_byte_offset = checkedMultiply(
            source_element_offset,
            source.elementSize(),
            "tensor copy source byte offset overflows size_t");
        const size_t destination_byte_offset = checkedMultiply(
            chunk_index,
            chunk_bytes,
            "tensor copy destination byte offset overflows size_t");

        if (on_cpu) {
            std::memcpy(
                destination.data() + destination_byte_offset,
                source.data() + source_byte_offset,
                chunk_bytes);
        } else {
            runtime->api()->memcpy_async(
                destination.data() + destination_byte_offset,
                source.data() + source_byte_offset,
                chunk_bytes,
                LLAISYS_MEMCPY_D2D,
                runtime->stream());
            ++queued_copies;
            // Avoid building an unbounded command queue for a heavily
            // permuted tensor whose contiguous chunk is only one element.
            if (queued_copies == 1024) {
                runtime->synchronize();
                queued_copies = 0;
            }
        }
    }

    if (runtime != nullptr && queued_copies != 0) {
        runtime->synchronize();
    }
}

} // namespace

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    CHECK_ARGUMENT(device >= 0, "tensor device id must not be negative");
    CHECK_ARGUMENT(
        device_type != LLAISYS_DEVICE_CPU || device == 0,
        "CPU tensor device id must be zero");

    std::vector<ptrdiff_t> strides = makeContiguousStrides(shape);
    TensorMeta meta{dtype, shape, strides};
    const size_t total_elems = checkedNumel(shape);
    const size_t dtype_size = utils::dsize(dtype);
    const size_t storage_size = checkedMultiply(
        total_elems,
        dtype_size,
        "tensor storage size overflows size_t");

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(storage_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(storage_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    std::byte *memory = _storage->memory();
    return memory == nullptr ? nullptr : memory + _offset;
}

const std::byte *Tensor::data() const {
    const std::byte *memory = _storage->memory();
    return memory == nullptr ? nullptr : memory + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return checkedNumel(_meta.shape);
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (shape.empty()) {
        if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
            std::cout << utils::cast<float>(data[0]) << std::endl;
        } else {
            std::cout << data[0] << std::endl;
        }
        return;
    }

    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->numel() == 0) {
        std::cout << "[]" << std::endl;
        return;
    }
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = this->to(LLAISYS_DEVICE_CPU, 0);
        debug_print(
            tmp_tensor->data(),
            tmp_tensor->shape(),
            tmp_tensor->strides(),
            tmp_tensor->dtype());
    }
}

bool Tensor::isContiguous() const {
    if (this->numel() == 0) {
        return true;
    }

    ptrdiff_t expected_stride = 1;

    for (size_t i = this->ndim(); i > 0; --i) {
        const size_t dim = i - 1;

        if (this->shape()[dim] != 1 && this->strides()[dim] != expected_stride) {
            return false;
        }
        if (this->shape()[dim] != 1) {
            if (expected_stride < 0
                || this->shape()[dim]
                       > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())
                           / static_cast<size_t>(expected_stride)) {
                return false;
            }
            expected_stride *= static_cast<ptrdiff_t>(this->shape()[dim]);
        }
    }

    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(
        order.size() == this->ndim(),
        "permute order size must match tensor dimensions");

    std::vector<bool> visited(this->ndim(), false);
    std::vector<size_t> new_shape;
    std::vector<ptrdiff_t> new_strides;

    for (size_t dim : order) {
        CHECK_ARGUMENT(
            dim < this->ndim(),
            "permute dimension is out of range");

        CHECK_ARGUMENT(
            !visited[dim],
            "permute dimensions must not repeat");

        visited[dim] = true;
        new_shape.push_back(this->shape()[dim]);
        new_strides.push_back(this->strides()[dim]);
    }
    TensorMeta new_meta{
        this->dtype(),
        new_shape,
        new_strides};
    return std::shared_ptr<Tensor>(
        new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    CHECK_ARGUMENT(
        checkedNumel(shape) == this->numel(),
        "view cannot change the number of elements");

    std::vector<ptrdiff_t> new_strides;
    CHECK_ARGUMENT(
        computeViewStrides(this->shape(), this->strides(), shape, new_strides),
        "view shape is incompatible with tensor strides");

    TensorMeta new_meta{
        this->dtype(),
        shape,
        std::move(new_strides),
    };

    return std::shared_ptr<Tensor>(
        new Tensor(std::move(new_meta), this->_storage, this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(
        dim < this->ndim(),
        "slice dimension is out of range");

    CHECK_ARGUMENT(
        start <= end,
        "slice start must not be greater than end");

    CHECK_ARGUMENT(
        end <= this->shape()[dim],
        "slice end is out of range");

    TensorMeta new_meta = this->_meta;
    new_meta.shape[dim] = end - start;

    CHECK_ARGUMENT(this->strides()[dim] >= 0, "negative tensor strides are not supported");
    const size_t offset_elements = checkedMultiply(
        start,
        static_cast<size_t>(this->strides()[dim]),
        "slice offset overflows size_t");
    const size_t offset_delta = checkedMultiply(
        offset_elements,
        this->elementSize(),
        "slice byte offset overflows size_t");
    size_t new_offset = checkedAdd(
        this->_offset,
        offset_delta,
        "slice byte offset overflows size_t");
    if (checkedNumel(new_meta.shape) == 0 && new_offset > this->_storage->size()) {
        // The storage offset of an empty tensor is not observable. Clamp it to
        // the one-past-the-end pointer so chained empty slices never construct
        // an invalid pointer beyond the allocation.
        new_offset = this->_storage->size();
    }
    CHECK_ARGUMENT(new_offset <= this->_storage->size(), "slice offset exceeds tensor storage");

    return std::shared_ptr<Tensor>(
        new Tensor(std::move(new_meta), this->_storage, new_offset));
}

void Tensor::load(const void *src_) {
    const size_t bytes = checkedMultiply(
        this->numel(),
        this->elementSize(),
        "tensor load byte count overflows size_t");
    if (bytes == 0) {
        return;
    }

    CHECK_ARGUMENT(src_ != nullptr, "tensor load source pointer must not be null");
    CHECK_ARGUMENT(this->isContiguous(), "loading into a non-contiguous tensor is not supported");
    CHECK_ARGUMENT(this->data() != nullptr, "tensor data pointer is null");

    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->memcpy_sync(
        this->data(),
        src_,
        bytes,
        this->deviceType() == LLAISYS_DEVICE_CPU
            ? LLAISYS_MEMCPY_H2H
            : LLAISYS_MEMCPY_H2D);
}

tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(
            new Tensor(this->_meta, this->_storage, this->_offset));
    }

    auto result = Tensor::create(
        this->shape(),
        this->dtype(),
        this->deviceType(),
        this->deviceId());
    copyLogicalToContiguous(*this, *result);
    return result;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    CHECK_ARGUMENT(
        checkedNumel(shape) == this->numel(),
        "reshape cannot change the number of elements");

    std::vector<ptrdiff_t> new_strides;
    if (computeViewStrides(this->shape(), this->strides(), shape, new_strides)) {
        TensorMeta new_meta{this->dtype(), shape, std::move(new_strides)};
        return std::shared_ptr<Tensor>(
            new Tensor(std::move(new_meta), this->_storage, this->_offset));
    }

    return this->contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    CHECK_ARGUMENT(device >= -1, "tensor device id must be -1 or non-negative");
    if (device < 0) {
        device = device_type == this->deviceType() ? this->deviceId() : 0;
    }

    if (device_type == this->deviceType() && device == this->deviceId()) {
        return std::shared_ptr<Tensor>(
            new Tensor(this->_meta, this->_storage, this->_offset));
    }

    auto source = this->contiguous();
    auto result = Tensor::create(
        this->shape(),
        this->dtype(),
        device_type,
        device);

    const size_t bytes = checkedMultiply(
        this->numel(),
        this->elementSize(),
        "tensor transfer byte count overflows size_t");
    if (bytes == 0) {
        return result;
    }

    CHECK_ARGUMENT(source->data() != nullptr && result->data() != nullptr, "tensor data pointer is null");

    if (source->deviceType() == LLAISYS_DEVICE_CPU
        && result->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(result->data(), source->data(), bytes);
    } else if (source->deviceType() == LLAISYS_DEVICE_CPU) {
        core::context().setDevice(result->deviceType(), result->deviceId());
        core::context().runtime().api()->memcpy_sync(
            result->data(),
            source->data(),
            bytes,
            LLAISYS_MEMCPY_H2D);
    } else if (result->deviceType() == LLAISYS_DEVICE_CPU) {
        core::context().setDevice(source->deviceType(), source->deviceId());
        core::context().runtime().synchronize();
        core::context().runtime().api()->memcpy_sync(
            result->data(),
            source->data(),
            bytes,
            LLAISYS_MEMCPY_D2H);
    } else {
        // Runtime APIs do not expose a peer-copy operation. Stage through host
        // memory so transfers between device IDs or accelerator types remain
        // correct without assuming peer access has been enabled.
        std::vector<std::byte> host_buffer(bytes);
        core::context().setDevice(source->deviceType(), source->deviceId());
        core::context().runtime().synchronize();
        core::context().runtime().api()->memcpy_sync(
            host_buffer.data(),
            source->data(),
            bytes,
            LLAISYS_MEMCPY_D2H);

        core::context().setDevice(result->deviceType(), result->deviceId());
        core::context().runtime().api()->memcpy_sync(
            result->data(),
            host_buffer.data(),
            bytes,
            LLAISYS_MEMCPY_H2D);
    }

    return result;
}

} // namespace llaisys
