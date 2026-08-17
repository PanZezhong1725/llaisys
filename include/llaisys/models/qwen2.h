#ifndef LLAISYS_QWEN2_MODEL_API_H
#define LLAISYS_QWEN2_MODEL_API_H

#include "../tensor.h"

__C {

typedef struct LlaisysQwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer;
    size_t hs;
    size_t nh;
    size_t nkvh;
    size_t dh;
    size_t di;
    size_t maxseq;
    size_t voc;
    float epsilon;
    float theta;
    int64_t end_token;
} LlaisysQwen2Meta;

typedef struct LlaisysQwen2Weights {
    llaisysTensor_t in_embed;
    llaisysTensor_t out_embed;
    llaisysTensor_t out_norm_w;
    llaisysTensor_t *attn_norm_w;
    llaisysTensor_t *attn_q_w;
    llaisysTensor_t *attn_q_b;
    llaisysTensor_t *attn_k_w;
    llaisysTensor_t *attn_k_b;
    llaisysTensor_t *attn_v_w;
    llaisysTensor_t *attn_v_b;
    llaisysTensor_t *attn_o_w;
    llaisysTensor_t *mlp_norm_w;
    llaisysTensor_t *mlp_gate_w;
    llaisysTensor_t *mlp_up_w;
    llaisysTensor_t *mlp_down_w;
} LlaisysQwen2Weights;

typedef struct LlaisysQwen2Model LlaisysQwen2Model;

__export LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice);
__export void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model);
__export LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model);
__export void llaisysQwen2ModelReset(LlaisysQwen2Model *model);
__export int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken);

}

#endif
