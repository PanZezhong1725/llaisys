#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../device/runtime_api.hpp"
#include "../../utils.hpp"


#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.hpp"
#endif
#ifdef ENABLE_SUDA_API
#include "suda/rope_suda.hpp"
#endif


#include <cmath>
#include <cstring>




namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");

    size_t seq_len = in->shape()[0];
    size_t num_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    // Compute cos and sin from pos_ids and theta
    // cos[i][j] = cos(pos_ids[i] / theta^(2j/head_dim))
    // sin[i][j] = sin(pos_ids[i] / theta^(2j/head_dim))
    size_t half_dim = head_dim / 2;
    std::vector<float> cos_buf(seq_len * head_dim);
    std::vector<float> sin_buf(seq_len * head_dim);

    // Read pos_ids as int64_t. For device tensors (e.g. NVIDIA), the data pointer
    // is a device pointer, so copy it to a host buffer first.
    std::vector<int64_t> pos_host(seq_len);
    const int64_t *pos_data = reinterpret_cast<const int64_t *>(pos_ids->data());
    if (pos_ids->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(pos_host.data(), pos_data, seq_len * sizeof(int64_t));
    } else {
        llaisys::device::getRuntimeAPI(pos_ids->deviceType())
            ->memcpy_sync(pos_host.data(), pos_data, seq_len * sizeof(int64_t),
                          LLAISYS_MEMCPY_D2H);
    }
    for (size_t i = 0; i < seq_len; i++) {
        // Torch uses float64 for position arithmetic; match it
        double pos = static_cast<double>(pos_host[i]);
        for (size_t j = 0; j < half_dim; j++) {
            // Compute freq matching Torch: positions / (theta ** (2*j/head_dim))
            double exponent = 2.0 * static_cast<double>(j) / static_cast<double>(head_dim);
            double denom = pow(static_cast<double>(theta), exponent);
            double freq = pos / denom;
            double c = cos(freq);
            double s = sin(freq);

            cos_buf[i * head_dim + j] = static_cast<float>(c);
            cos_buf[i * head_dim + j + half_dim] = static_cast<float>(c);
            sin_buf[i * head_dim + j] = static_cast<float>(s);
            sin_buf[i * head_dim + j + half_dim] = static_cast<float>(s);
        }
    }






    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(),
                         reinterpret_cast<const std::byte *>(cos_buf.data()),
                         reinterpret_cast<const std::byte *>(sin_buf.data()),
                         out->dtype(), seq_len, num_heads, head_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(),
                         reinterpret_cast<const std::byte *>(cos_buf.data()),
                         reinterpret_cast<const std::byte *>(sin_buf.data()),
                         out->dtype(), seq_len, num_heads, head_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::rope(out->data(), in->data(),
                     reinterpret_cast<const int64_t *>(pos_ids->data()), theta,
                     out->dtype(), seq_len, num_heads, head_dim);
        return;
#endif
#ifdef ENABLE_SUDA_API
    case LLAISYS_DEVICE_SUDA:
        suda::rope(out->data(), in->data(),
                   reinterpret_cast<const int64_t *>(pos_ids->data()), theta,
                   out->dtype(), seq_len, num_heads, head_dim);
        return;
#endif


    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops


