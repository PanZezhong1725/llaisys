#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif

namespace llaisys::ops {

void self_attention(tensor_t output, tensor_t query, tensor_t key, tensor_t value, float scale) {
    CHECK_SAME_DEVICE(output, query, key, value);
    CHECK_SAME_DTYPE(output->dtype(), query->dtype(), key->dtype(), value->dtype());
    CHECK_ARGUMENT(output->ndim() == 3 && query->ndim() == 3 && key->ndim() == 3 && value->ndim() == 3, "self_attention expects rank-three tensors");
    CHECK_ARGUMENT(key->shape()[0] == value->shape()[0] && key->shape()[1] == value->shape()[1], "key/value layout mismatch");
    CHECK_ARGUMENT(query->shape()[2] == key->shape()[2], "query/key width mismatch");
    CHECK_ARGUMENT(output->shape()[0] == query->shape()[0] && output->shape()[1] == query->shape()[1] && output->shape()[2] == value->shape()[2], "attention output shape mismatch");
    CHECK_ARGUMENT(key->shape()[0] >= query->shape()[0], "key sequence must cover query sequence");
    CHECK_ARGUMENT(key->shape()[1] != 0 && query->shape()[1] % key->shape()[1] == 0, "invalid grouped-query head count");
    CHECK_ARGUMENT(output->isContiguous() && query->isContiguous() && key->isContiguous() && value->isContiguous(), "self_attention requires contiguous tensors");

    core::context().setDevice(output->deviceType(), output->deviceId());
    const size_t query_length = query->shape()[0];
    const size_t key_length = key->shape()[0];
    const size_t query_heads = query->shape()[1];
    const size_t kv_heads = key->shape()[1];
    const size_t query_key_width = query->shape()[2];
    const size_t value_width = value->shape()[2];
    if (output->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(output->data(), query->data(), key->data(), value->data(), output->dtype(), query_length, key_length, query_heads, kv_heads, query_key_width, value_width, scale);
    }
#ifdef ENABLE_NVIDIA_API
    if (output->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::self_attention(output->data(), query->data(), key->data(), value->data(), output->dtype(), query_length, key_length, query_heads, kv_heads, query_key_width, value_width, scale, core::context().runtime().stream());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
