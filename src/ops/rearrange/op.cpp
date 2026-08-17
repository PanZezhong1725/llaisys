#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cstring>
#include <vector>

namespace llaisys::ops {

void rearrange(tensor_t output, tensor_t input) {
    CHECK_SAME_DEVICE(output, input);
    CHECK_SAME_SHAPE(output->shape(), input->shape());
    CHECK_SAME_DTYPE(output->dtype(), input->dtype());
    CHECK_ARGUMENT(output->isContiguous(), "rearrange output must be contiguous");

    const size_t bytes = input->numel() * input->elementSize();
    if (bytes == 0) {
        return;
    }

    core::context().setDevice(output->deviceType(), output->deviceId());
    if (input->isContiguous()) {
        core::context().runtime().api()->memcpy_sync(
            output->data(), input->data(), bytes,
            output->deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_D2D);
        return;
    }

    CHECK_ARGUMENT(output->deviceType() == LLAISYS_DEVICE_CPU, "non-contiguous GPU rearrange is not exposed by this operator");
    std::vector<size_t> coordinate(input->ndim(), 0);
    for (size_t linear = 0; linear < input->numel(); ++linear) {
        ptrdiff_t source_offset = 0;
        size_t remainder = linear;
        for (size_t dim = input->ndim(); dim-- > 0;) {
            coordinate[dim] = remainder % input->shape()[dim];
            remainder /= input->shape()[dim];
            source_offset += static_cast<ptrdiff_t>(coordinate[dim]) * input->strides()[dim];
        }
        std::memcpy(
            output->data() + linear * input->elementSize(),
            input->data() + source_offset * static_cast<ptrdiff_t>(input->elementSize()),
            input->elementSize());
    }
}

} // namespace llaisys::ops
