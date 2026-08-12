#include "linear_cpu.hpp"

#include "../../cpu/cpu_utils.hpp"

namespace {

template <typename T>
void linear_impl(std::byte *out,
                 const std::byte *in,
                 const std::byte *weight,
                 const std::byte *bias,
                 size_t batch_size,
                 size_t input_size,
                 size_t output_size) {
    const auto *input = reinterpret_cast<const T *>(in);
    const auto *weights = reinterpret_cast<const T *>(weight);
    const auto *bias_values = reinterpret_cast<const T *>(bias);
    auto *output = reinterpret_cast<T *>(out);

    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t out_feature = 0; out_feature < output_size; ++out_feature) {
            float sum = bias_values == nullptr ? 0.0F : llaisys::ops::cpu::to_float(bias_values[out_feature]);
            for (size_t in_feature = 0; in_feature < input_size; ++in_feature) {
                const float input_value = llaisys::ops::cpu::to_float(input[batch * input_size + in_feature]);
                const float weight_value = llaisys::ops::cpu::to_float(weights[out_feature * input_size + in_feature]);
                sum += input_value * weight_value;
            }
            output[batch * output_size + out_feature] = llaisys::ops::cpu::from_float<T>(sum);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {

void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t dtype,
            size_t batch_size,
            size_t input_size,
            size_t output_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_impl<float>(out, in, weight, bias, batch_size, input_size, output_size);
    case LLAISYS_DTYPE_F16:
        return linear_impl<llaisys::fp16_t>(out, in, weight, bias, batch_size, input_size, output_size);
    case LLAISYS_DTYPE_BF16:
        return linear_impl<llaisys::bf16_t>(out, in, weight, bias, batch_size, input_size, output_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::cpu
