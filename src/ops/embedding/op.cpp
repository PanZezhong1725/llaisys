#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

namespace llaisys::ops {

void embedding(tensor_t output, tensor_t indices, tensor_t table) {
    CHECK_SAME_DEVICE(output, indices, table);
    CHECK_ARGUMENT(indices->dtype() == LLAISYS_DTYPE_I64, "embedding indices must be int64");
    CHECK_ARGUMENT(indices->ndim() == 1 && table->ndim() == 2 && output->ndim() == 2, "embedding expects [N], [V,D], [N,D]");
    CHECK_ARGUMENT(output->shape()[0] == indices->shape()[0] && output->shape()[1] == table->shape()[1], "embedding output shape mismatch");
    CHECK_ARGUMENT(output->dtype() == table->dtype(), "embedding dtype mismatch");
    CHECK_ARGUMENT(output->isContiguous() && indices->isContiguous() && table->isContiguous(), "embedding requires contiguous tensors");

    core::context().setDevice(output->deviceType(), output->deviceId());
    const size_t count = indices->shape()[0];
    const size_t rows = table->shape()[0];
    const size_t width = table->shape()[1];
    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(output->data(), indices->data(), table->data(), output->dtype(), count, rows, width);
    }
#ifdef ENABLE_NVIDIA_API
    if (output->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::embedding(output->data(), indices->data(), table->data(), output->dtype(), count, rows, width, core::context().runtime().stream());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
