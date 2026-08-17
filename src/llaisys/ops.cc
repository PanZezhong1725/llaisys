#include "llaisys/ops.h"

#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

namespace {
llaisys::tensor_t tensorOf(llaisysTensor_t handle) {
    return handle == nullptr ? nullptr : handle->tensor;
}
} // namespace

__C {

void llaisysAdd(llaisysTensor_t output, llaisysTensor_t left, llaisysTensor_t right) {
    llaisys::ops::add(tensorOf(output), tensorOf(left), tensorOf(right));
}

void llaisysArgmax(llaisysTensor_t index, llaisysTensor_t value, llaisysTensor_t input) {
    llaisys::ops::argmax(tensorOf(index), tensorOf(value), tensorOf(input));
}

void llaisysEmbedding(llaisysTensor_t output, llaisysTensor_t indices, llaisysTensor_t table) {
    llaisys::ops::embedding(tensorOf(output), tensorOf(indices), tensorOf(table));
}

void llaisysLinear(llaisysTensor_t output, llaisysTensor_t input, llaisysTensor_t weight, llaisysTensor_t bias) {
    llaisys::ops::linear(tensorOf(output), tensorOf(input), tensorOf(weight), tensorOf(bias));
}

void llaisysRearrange(llaisysTensor_t output, llaisysTensor_t input) {
    llaisys::ops::rearrange(tensorOf(output), tensorOf(input));
}

void llaisysRmsNorm(llaisysTensor_t output, llaisysTensor_t input, llaisysTensor_t weight, float epsilon) {
    llaisys::ops::rms_norm(tensorOf(output), tensorOf(input), tensorOf(weight), epsilon);
}

void llaisysROPE(llaisysTensor_t output, llaisysTensor_t input, llaisysTensor_t positions, float theta) {
    llaisys::ops::rope(tensorOf(output), tensorOf(input), tensorOf(positions), theta);
}

void llaisysSelfAttention(llaisysTensor_t output, llaisysTensor_t query, llaisysTensor_t key, llaisysTensor_t value, float scale) {
    llaisys::ops::self_attention(tensorOf(output), tensorOf(query), tensorOf(key), tensorOf(value), scale);
}

void llaisysSwiGLU(llaisysTensor_t output, llaisysTensor_t gate, llaisysTensor_t up) {
    llaisys::ops::swiglu(tensorOf(output), tensorOf(gate), tensorOf(up));
}

}
