#pragma once

#include <cstdint>

namespace llaisys::device::metax {

// Keep storage types independent from compiler-specific FP16/BF16 class
// names. MXCC operates on float values while tensors retain their 16-bit ABI.
struct alignas(2) fp16_t {
    std::uint16_t bits;
};

struct alignas(2) bf16_t {
    std::uint16_t bits;
};

union FloatBits {
    float value;
    std::uint32_t bits;
};

template <typename T>
__device__ inline float to_float(T value);

template <>
__device__ inline float to_float<fp16_t>(fp16_t value) {
    const std::uint32_t sign = (value.bits & 0x8000u) << 16;
    std::uint32_t exponent = (value.bits >> 10) & 0x1fu;
    std::uint32_t mantissa = value.bits & 0x03ffu;
    std::uint32_t result;

    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            exponent = 113;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            result = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }

    FloatBits converted;
    converted.bits = result;
    return converted.value;
}

__device__ inline fp16_t fp16_from_float(float value) {
    FloatBits source;
    source.value = value;

    const std::uint32_t sign = (source.bits >> 16) & 0x8000u;
    const std::uint32_t magnitude = source.bits & 0x7fffffffu;
    fp16_t result{};

    if (magnitude >= 0x7f800000u) {
        const std::uint16_t payload = magnitude > 0x7f800000u
            ? static_cast<std::uint16_t>((magnitude >> 13) & 0x03ffu) | 1u
            : 0u;
        result.bits = static_cast<std::uint16_t>(sign | 0x7c00u | payload);
        return result;
    }

    const int exponent = static_cast<int>((magnitude >> 23) & 0xffu) - 127;
    std::uint32_t mantissa = magnitude & 0x7fffffu;

    if (exponent > 15) {
        result.bits = static_cast<std::uint16_t>(sign | 0x7c00u);
    } else if (exponent >= -14) {
        std::uint32_t rounded = mantissa + 0x0fffu + ((mantissa >> 13) & 1u);
        std::uint32_t half_exp = static_cast<std::uint32_t>(exponent + 15);
        if (rounded & 0x800000u) {
            rounded = 0;
            ++half_exp;
        }
        result.bits = static_cast<std::uint16_t>(
            sign | (half_exp << 10) | (rounded >> 13)
        );
    } else if (exponent >= -24) {
        mantissa |= 0x800000u;
        const int shift = -exponent - 1;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t rounded = mantissa + halfway - 1u
            + ((mantissa >> shift) & 1u);
        result.bits = static_cast<std::uint16_t>(sign | (rounded >> shift));
    } else {
        result.bits = static_cast<std::uint16_t>(sign);
    }

    return result;
}

template <>
__device__ inline float to_float<bf16_t>(bf16_t value) {
    FloatBits converted;
    converted.bits = static_cast<std::uint32_t>(value.bits) << 16;
    return converted.value;
}

__device__ inline bf16_t bf16_from_float(float value) {
    FloatBits source;
    source.value = value;
    const std::uint32_t rounding = 0x7fffu + ((source.bits >> 16) & 1u);
    return bf16_t{static_cast<std::uint16_t>((source.bits + rounding) >> 16)};
}

template <typename T>
__device__ inline T from_float(float value);

template <>
__device__ inline float from_float<float>(float value) {
    return value;
}

template <>
__device__ inline fp16_t from_float<fp16_t>(float value) {
    return fp16_from_float(value);
}

template <>
__device__ inline bf16_t from_float<bf16_t>(float value) {
    return bf16_from_float(value);
}

template <>
__device__ inline float to_float<float>(float value) {
    return value;
}

} // namespace llaisys::device::metax
