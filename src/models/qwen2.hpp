#pragma once

#include "../tensor/tensor.hpp"

#include <vector>

namespace llaisys::models {

struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

struct Qwen2Weights {
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

class Qwen2Model {
private:
    Qwen2Meta _meta;
    Qwen2Weights _weights;
    llaisysDeviceType_t _device_type;
    int _device_id;
    
    // KV Cache
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;
    size_t _cache_len;

public:
    Qwen2Model(const Qwen2Meta &meta, llaisysDeviceType_t device_type, int device_id);
    ~Qwen2Model() = default;
    
    Qwen2Weights &weights();
    int64_t infer(const std::vector<int64_t> &token_ids);
    
private:
    tensor_t forward(const std::vector<int64_t> &token_ids);
    tensor_t embedding(const std::vector<int64_t> &token_ids);
    tensor_t attention(size_t layer_idx, tensor_t hidden, size_t pos_offset);
    tensor_t mlp(size_t layer_idx, tensor_t hidden);
    tensor_t rms_norm(tensor_t x, tensor_t weight, float eps);
};

} // namespace llaisys::models
