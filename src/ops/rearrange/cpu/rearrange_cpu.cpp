#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, size_t numel) {
    size_t element_size = utils::dsize(type);
    std::memcpy(out, in, numel * element_size);
}
} // namespace llaisys::ops::cpu
