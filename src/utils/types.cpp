#include "types.hpp"

#include <cstring>

namespace llaisys::utils {
float _f16_to_f32(fp16_t val) {
    uint16_t h = val._v;
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t f32;
    if (exponent == 31) {
        if (mantissa != 0) {
            f32 = sign | 0x7F800000 | (mantissa << 13);
        } else {
            f32 = sign | 0x7F800000;
        }
    } else if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign;
        } else {
            exponent = -14;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32 = sign | ((exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        f32 = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

fp16_t _f32_to_f16(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));

    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t exponent_bits = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;

    // Preserve infinities and emit a quiet half-precision NaN. Keeping at
    // least one payload bit prevents a float32 NaN with a tiny payload from
    // being truncated into infinity.
    if (exponent_bits == 0xffu) {
        if (mantissa == 0) {
            return fp16_t{static_cast<uint16_t>(sign | 0x7c00u)};
        }
        uint16_t payload = static_cast<uint16_t>(mantissa >> 13);
        payload = static_cast<uint16_t>((payload | 0x0200u) & 0x03ffu);
        return fp16_t{static_cast<uint16_t>(sign | 0x7c00u | payload)};
    }

    // Float32 subnormals are far below the representable half range.
    if (exponent_bits == 0) {
        return fp16_t{sign};
    }

    const int32_t exponent = static_cast<int32_t>(exponent_bits) - 127;
    if (exponent > 15) {
        return fp16_t{static_cast<uint16_t>(sign | 0x7c00u)};
    }

    if (exponent >= -14) {
        uint32_t rounded = mantissa;
        // Round the discarded 13 bits to nearest, ties to even.
        rounded += 0x0fffu + ((rounded >> 13) & 1u);
        uint32_t half_exponent = static_cast<uint32_t>(exponent + 15);
        if ((rounded & 0x800000u) != 0) {
            rounded = 0;
            ++half_exponent;
            if (half_exponent >= 31) {
                return fp16_t{static_cast<uint16_t>(sign | 0x7c00u)};
            }
        }
        return fp16_t{static_cast<uint16_t>(
            sign | (half_exponent << 10) | ((rounded >> 13) & 0x03ffu))};
    }

    if (exponent < -25) {
        return fp16_t{sign};
    }

    // Half subnormal. The implicit float32 leading one participates in the
    // rounding; exponent == -25 is the exact zero/min-subnormal tie case.
    const uint32_t significand = mantissa | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(-exponent - 1);
    uint32_t half_mantissa = significand >> shift;
    const uint32_t remainder_mask = (uint32_t{1} << shift) - 1u;
    const uint32_t remainder = significand & remainder_mask;
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    if (remainder > halfway
        || (remainder == halfway && (half_mantissa & 1u) != 0)) {
        ++half_mantissa;
    }

    // A rounded subnormal may naturally carry into the minimum normal value
    // (0x0400), so do not mask the result down to ten bits here.
    return fp16_t{static_cast<uint16_t>(sign | half_mantissa)};
}

float _bf16_to_f32(bf16_t val) {
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;

    float out;
    std::memcpy(&out, &bits32, sizeof(out));
    return out;
}

bf16_t _f32_to_bf16(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));

    const uint32_t rounding_bias = 0x00007FFF + // 0111 1111 1111 1111
                                   ((bits32 >> 16) & 1);

    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);

    return bf16_t{bf16_bits};
}
} // namespace llaisys::utils
