#include "op.hpp"

#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif
#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/argmax_iluvatar.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/argmax_metax.hpp"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    // 暂时假设 1D
    CHECK_ARGUMENT(
        vals->shape().size() == 1, 
        "Argmax: vals must be a 1D tensor.");
    
    CHECK_ARGUMENT(
        vals->numel() > 0, 
        "Argmax: vals must not be empty."
    );
    
    // max_idx  max_val 必须是 1 维单数字
    CHECK_ARGUMENT(
        max_idx->shape().size() == 1 && max_idx->numel() == 1,
        "Argmax: max_idx must be a 1D tensor containing one element."
    );
    CHECK_ARGUMENT(
        max_val->shape().size() == 1 && max_val->numel() == 1,
        "Argmax: max_val must be a 1D tensor containing one element."
    );

    // idx 是整数
    // PyTorch 的 argmax 返回值默认也是 torch.int64
    CHECK_ARGUMENT(
        max_idx->dtype() ==  LLAISYS_DTYPE_I64,
        "Argmax: max_idx must have int64 dtype."
    );

    CHECK_ARGUMENT(
        vals->dtype() == max_val->dtype(),
        "Argmax: max_val and vals must have the same dtype."
    );

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel()
        );
    }

    llaisys::core::context().setDevice( vals->deviceType(),  vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel()
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel(),
            llaisys::core::context().runtime().stream()
        );
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel(),
            llaisys::core::context().runtime().stream()
        );
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel(),
            llaisys::core::context().runtime().stream()
        );
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
