#pragma once

#include "llaisys/qwen2.h"
#include "llaisys_tensor.hpp"

#include "../tensor/tensor.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <cstring>

namespace llaisys {
namespace models {

// Qwen2 model configuration constants for DeepSeek-R1-Distill-Qwen-1.5B
struct Qwen2Config {
    size_t hidden_size = 1536;
    size_t intermediate_size = 8960;
    size_t num_attention_heads = 12;
    size_t num_key_value_heads = 2;
    size_t num_hidden_layers = 28;
    size_t vocab_size = 151936;
    size_t max_position_embeddings = 131072;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
};

class Qwen2Model {
public:
    Qwen2Model(llaisysDeviceType_t device_type, int device_id);
    ~Qwen2Model() = default;

    // Load a weight tensor by parameter name
    void loadWeight(const std::string &name, tensor_t weight);

    // Forward pass: input_ids [seq_len], output logits [vocab_size]
    void forward(tensor_t input_ids, tensor_t output_logits);

    // Reset KV cache
    void resetKV();

private:
    Qwen2Config config;
    llaisysDeviceType_t device_type;
    int device_id;

    // Weight storage: maps parameter name -> tensor
    std::unordered_map<std::string, tensor_t> weights;

    // KV cache: [num_layers][2] where 0=key, 1=value
    // Each cache tensor: [num_kv_heads, max_seq_len, head_dim]
    struct KVCache {
        tensor_t key;
        tensor_t value;
        size_t current_seq_len;
    };
    std::vector<KVCache> kv_caches;
    size_t max_cache_len;

    // Helper to get weight tensor
    tensor_t getWeight(const std::string &name) const;

    // Embedding lookup
    void embedding(tensor_t hidden_states, tensor_t input_ids);

    // Transformer layer
    void transformerLayer(
        tensor_t hidden_states,
        size_t layer_idx,
        size_t start_pos,
        size_t seq_len);

    // Final RMS norm + lm_head
    void finalNormAndHead(tensor_t hidden_states, tensor_t logits);

    // Create a tensor on the model's device
    tensor_t createTensor(const std::vector<size_t> &shape, llaisysDataType_t dtype) const;
};

} // namespace models
} // namespace llaisys
