#pragma once

#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"

#include <vector>

namespace llaisys::models::qwen2 {

struct Qwen2LayerWeights {
    tensor_t attn_norm_w;
    tensor_t attn_q_w;
    tensor_t attn_q_b;
    tensor_t attn_k_w;
    tensor_t attn_k_b;
    tensor_t attn_v_w;
    tensor_t attn_v_b;
    tensor_t attn_o_w;
    tensor_t mlp_norm_w;
    tensor_t mlp_gate_w;
    tensor_t mlp_up_w;
    tensor_t mlp_down_w;
};

struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<Qwen2LayerWeights> layers;
};

struct Qwen2KVCache {
    tensor_t k;
    tensor_t v;
    size_t length{0};
    size_t capacity{0};
};

class Qwen2Model final {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, std::vector<int> device_ids);

    const LlaisysQwen2Meta &meta() const;
    llaisysDeviceType_t deviceType() const;
    int primaryDeviceId() const;
    Qwen2Weights &weights();
    std::vector<Qwen2KVCache> &kvCaches();
    void resetCache();
    int64_t infer(const int64_t *token_ids, size_t ntoken);

private:
    tensor_t embedTokens(const int64_t *token_ids, size_t ntoken) const;
    tensor_t createPositionIds(size_t start, size_t ntoken) const;
    int64_t selectNextToken(tensor_t hidden) const;
    tensor_t forwardAttention(tensor_t hidden, tensor_t position_ids, const Qwen2LayerWeights &weights, Qwen2KVCache &cache);
    tensor_t forwardMlp(tensor_t hidden, const Qwen2LayerWeights &weights) const;
    tensor_t forwardLayer(tensor_t hidden, tensor_t position_ids, const Qwen2LayerWeights &weights, Qwen2KVCache &cache);
    void ensureCacheCapacity(Qwen2KVCache &cache, size_t required);
    void appendKvCache(Qwen2KVCache &cache, tensor_t new_key, tensor_t new_value);

    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    std::vector<int> _device_ids;
    Qwen2Weights _weights;
    std::vector<Qwen2KVCache> _kv_caches;
};

} // namespace llaisys::models::qwen2
