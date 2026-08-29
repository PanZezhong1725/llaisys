#pragma once

#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"

#include <vector>

namespace llaisys::models::qwen2 {

// Weights held as llaisys tensors, in the same order as LlaisysQwen2Weights so the C
// layer can hand raw handles to Python for loading.
struct Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<tensor_t> attn_norm_w;
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;
};

class Qwen2 {
public:
    Qwen2(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id);

    Weights &weights() { return _weights; }

    // Appends ntoken tokens to the KV cache and returns the argmax next token.
    int64_t infer(const int64_t *token_ids, size_t ntoken);

    void resetCache() { _past_len = 0; }

private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;
    Weights _weights;

    // Per-layer KV cache, shape [maxseq, nkvh, dh]; _past_len rows are valid.
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;
    size_t _past_len = 0;

    tensor_t create(const std::vector<size_t> &shape);
};

} // namespace llaisys::models::qwen2
