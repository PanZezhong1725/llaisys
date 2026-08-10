#ifndef LLAISYS_QWEN2_H
#define LLAISYS_QWEN2_H

#include "../llaisys.h"
#include "tensor.h"

__C {
    // Opaque handle for Qwen2 model
    typedef struct LlaisysQwen2Model *llaisysQwen2Model_t;

    // Create a Qwen2 model
    __export llaisysQwen2Model_t qwen2Create(
        llaisysDeviceType_t device_type,
        int device_id);

    // Destroy a Qwen2 model
    __export void qwen2Destroy(
        llaisysQwen2Model_t model);

    // Load a weight tensor into the model by parameter name
    __export void qwen2LoadWeight(
        llaisysQwen2Model_t model,
        const char *name,
        llaisysTensor_t weight);

    // Forward pass: given input token IDs, produce logits
    // input_ids: [seq_len] int64 tensor on CPU
    // output_logits: [vocab_size] float32 tensor on CPU (pre-allocated)
    __export void qwen2Forward(
        llaisysQwen2Model_t model,
        llaisysTensor_t input_ids,
        llaisysTensor_t output_logits);

    // Reset KV cache (for new generation)
    __export void qwen2ResetKV(
        llaisysQwen2Model_t model);
}

#endif // LLAISYS_QWEN2_H
