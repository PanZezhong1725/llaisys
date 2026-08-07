#include "op.hpp"
#include "../../utils.hpp"
#include <cstring>
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/nvidia_ops.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_ARGUMENT(out->deviceType() == LLAISYS_DEVICE_CPU || out->deviceType() == LLAISYS_DEVICE_NVIDIA, "unsupported embedding device");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "embedding indices must be int64");
    CHECK_ARGUMENT(index->ndim() == 1 && weight->ndim() == 2 && out->ndim() == 2, "invalid embedding ranks");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0] && out->shape()[1] == weight->shape()[1], "invalid embedding output shape");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "embedding dtype mismatch");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "embedding requires contiguous tensors");
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
#ifdef ENABLE_NVIDIA_API
        return nvidia::embedding(out->data(), index->data(), weight->data(), out->dtype(), index->shape()[0], weight->shape()[1], weight->shape()[0]);
#else
        EXCEPTION_UNSUPPORTED_DEVICE;
#endif
    }
    const auto *indices = reinterpret_cast<const int64_t *>(index->data());
    size_t row_bytes = weight->shape()[1] * weight->elementSize();
    for (size_t i = 0; i < index->shape()[0]; i++) {
        CHECK_ARGUMENT(indices[i] >= 0 && static_cast<size_t>(indices[i]) < weight->shape()[0], "embedding index out of range");
        std::memcpy(out->data() + i * row_bytes,
                    weight->data() + static_cast<size_t>(indices[i]) * row_bytes, row_bytes);
    }
}
} // namespace llaisys::ops
