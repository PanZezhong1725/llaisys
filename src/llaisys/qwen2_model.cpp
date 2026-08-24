#include "qwen2_model.hpp"

#include "../core/llaisys_core.hpp"
#include "../device/runtime_api.hpp"
#include "../ops/add/op.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>

namespace llaisys {
namespace models {

Qwen2Model::Qwen2Model(llaisysDeviceType_t device_type, int device_id)
    : device_type(device_type), device_id(device_id) {
    size_t head_dim = config.hidden_size / config.num_attention_heads;
    max_cache_len = config.max_position_embeddings;

    kv_caches.resize(config.num_hidden_layers);
    for (size_t i = 0; i < config.num_hidden_layers; i++) {
        kv_caches[i].key = createTensor(
            {max_cache_len, config.num_key_value_heads, head_dim},
            LLAISYS_DTYPE_F32);
        kv_caches[i].value = createTensor(
            {max_cache_len, config.num_key_value_heads, head_dim},
            LLAISYS_DTYPE_F32);
        kv_caches[i].current_seq_len = 0;
    }
}

tensor_t Qwen2Model::createTensor(const std::vector<size_t> &shape, llaisysDataType_t dtype) const {
    return Tensor::create(shape, dtype, device_type, device_id);
}

tensor_t Qwen2Model::getWeight(const std::string &name) const {
    auto it = weights.find(name);
    if (it == weights.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

void Qwen2Model::loadWeight(const std::string &name, tensor_t weight) {
    weights[name] = weight;
}

void Qwen2Model::resetKV() {
    for (size_t i = 0; i < config.num_hidden_layers; i++) {
        kv_caches[i].current_seq_len = 0;
    }
}

void Qwen2Model::embedding(tensor_t hidden_states, tensor_t input_ids) {
    auto weight = getWeight("model.embed_tokens.weight");
    ops::embedding(hidden_states, input_ids, weight);
}

void Qwen2Model::transformerLayer(
    tensor_t hidden_states,
    size_t layer_idx,
    size_t start_pos,
    size_t seq_len) {

    size_t head_dim = config.hidden_size / config.num_attention_heads;
    size_t num_heads = config.num_attention_heads;
    size_t num_kv_heads = config.num_key_value_heads;
    auto no_bias = createTensor({0}, LLAISYS_DTYPE_F32);

    auto attn_norm_weight = getWeight("model.layers." + std::to_string(layer_idx) + ".input_layernorm.weight");
    auto attn_norm_hidden = createTensor({seq_len, config.hidden_size}, LLAISYS_DTYPE_F32);
    ops::rms_norm(attn_norm_hidden, hidden_states, attn_norm_weight, config.rms_norm_eps);

    auto q_proj_weight = getWeight("model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.weight");
    auto k_proj_weight = getWeight("model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.weight");
    auto v_proj_weight = getWeight("model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.weight");

    auto q_proj_bias = getWeight("model.layers." + std::to_string(layer_idx) + ".self_attn.q_proj.bias");
    auto k_proj_bias = getWeight("model.layers." + std::to_string(layer_idx) + ".self_attn.k_proj.bias");
    auto v_proj_bias = getWeight("model.layers." + std::to_string(layer_idx) + ".self_attn.v_proj.bias");

    auto q = createTensor({seq_len, config.hidden_size}, LLAISYS_DTYPE_F32);
    auto k = createTensor({seq_len, num_kv_heads * head_dim}, LLAISYS_DTYPE_F32);
    auto v = createTensor({seq_len, num_kv_heads * head_dim}, LLAISYS_DTYPE_F32);

    ops::linear(q, attn_norm_hidden, q_proj_weight, q_proj_bias);
    ops::linear(k, attn_norm_hidden, k_proj_weight, k_proj_bias);
    ops::linear(v, attn_norm_hidden, v_proj_weight, v_proj_bias);

    auto q_rope = q->view({seq_len, num_heads, head_dim});
    auto k_rope = k->view({seq_len, num_kv_heads, head_dim});
    auto v_contiguous = v->view({seq_len, num_kv_heads, head_dim});

    auto pos_ids_data = new int64_t[seq_len];
    for (size_t i = 0; i < seq_len; i++) {
        pos_ids_data[i] = static_cast<int64_t>(start_pos + i);
    }
    auto pos_ids = createTensor({seq_len}, LLAISYS_DTYPE_I64);
    pos_ids->load(pos_ids_data);
    delete[] pos_ids_data;

    ops::rope(q_rope, q_rope, pos_ids, config.rope_theta);
    ops::rope(k_rope, k_rope, pos_ids, config.rope_theta);

    // Update KV cache
    auto &kv_cache = kv_caches[layer_idx];
    size_t cache_pos = kv_cache.current_seq_len;

    for (size_t h = 0; h < num_kv_heads; h++) {
        // FIXED: source offset is h * head_dim, not h * seq_len * head_dim
        float *k_src = reinterpret_cast<float *>(k_rope->data()) + h * head_dim;
        float *k_dst = reinterpret_cast<float *>(kv_cache.key->data()) + cache_pos * num_kv_heads * head_dim + h * head_dim;
        for (size_t s = 0; s < seq_len; s++) {
            llaisys::device::getRuntimeAPI(device_type)
                ->memcpy_sync(k_dst + s * num_kv_heads * head_dim,
                              k_src + s * num_kv_heads * head_dim,
                              head_dim * sizeof(float), LLAISYS_MEMCPY_D2D);
        }

        float *v_src = reinterpret_cast<float *>(v_contiguous->data()) + h * head_dim;
        float *v_dst = reinterpret_cast<float *>(kv_cache.value->data()) + cache_pos * num_kv_heads * head_dim + h * head_dim;
        for (size_t s = 0; s < seq_len; s++) {
            llaisys::device::getRuntimeAPI(device_type)
                ->memcpy_sync(v_dst + s * num_kv_heads * head_dim,
                              v_src + s * num_kv_heads * head_dim,
                              head_dim * sizeof(float), LLAISYS_MEMCPY_D2D);
        }
    }
    kv_cache.current_seq_len = cache_pos + seq_len;

    // Self-attention
    size_t kv_len = kv_cache.current_seq_len;
    auto kv_key = createTensor({kv_len, num_kv_heads, head_dim}, LLAISYS_DTYPE_F32);
    auto kv_value = createTensor({kv_len, num_kv_heads, head_dim}, LLAISYS_DTYPE_F32);
    for (size_t h = 0; h < num_kv_heads; h++) {
        for (size_t t = 0; t < kv_len; t++) {
            float *k_src_cache = reinterpret_cast<float *>(kv_cache.key->data()) + t * num_kv_heads * head_dim + h * head_dim;
            float *k_dst_cache = reinterpret_cast<float *>(kv_key->data()) + t * num_kv_heads * head_dim + h * head_dim;
            llaisys::device::getRuntimeAPI(device_type)
                ->memcpy_sync(k_dst_cache, k_src_cache,
                              head_dim * sizeof(float), LLAISYS_MEMCPY_D2D);

            float *v_src_cache = reinterpret_cast<float *>(kv_cache.value->data()) + t * num_kv_heads * head_dim + h * head_dim;
            float *v_dst_cache = reinterpret_cast<float *>(kv_value->data()) + t * num_kv_heads * head_dim + h * head_dim;
            llaisys::device::getRuntimeAPI(device_type)
                ->memcpy_sync(v_dst_cache, v_src_cache,
                              head_dim * sizeof(float), LLAISYS_MEMCPY_D2D);
        }
    }

    float scale = 1.0f / sqrt(static_cast<float>(head_dim));
    auto attn_output = createTensor({seq_len, num_heads, head_dim}, LLAISYS_DTYPE_F32);
    ops::self_attention(attn_output, q_rope, kv_key, kv_value, scale);

    auto attn_flat = attn_output->view({seq_len, config.hidden_size});

    auto o_proj_weight = getWeight("model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj.weight");
    auto attn_out = createTensor({seq_len, config.hidden_size}, LLAISYS_DTYPE_F32);
    ops::linear(attn_out, attn_flat, o_proj_weight, no_bias);

    ops::add(hidden_states, hidden_states, attn_out);

    // FFN
    auto ffn_norm_weight = getWeight("model.layers." + std::to_string(layer_idx) + ".post_attention_layernorm.weight");
    auto ffn_norm_hidden = createTensor({seq_len, config.hidden_size}, LLAISYS_DTYPE_F32);
    ops::rms_norm(ffn_norm_hidden, hidden_states, ffn_norm_weight, config.rms_norm_eps);

    auto gate_proj_weight = getWeight("model.layers." + std::to_string(layer_idx) + ".mlp.gate_proj.weight");
    auto up_proj_weight = getWeight("model.layers." + std::to_string(layer_idx) + ".mlp.up_proj.weight");

    auto gate = createTensor({seq_len, config.intermediate_size}, LLAISYS_DTYPE_F32);
    auto up = createTensor({seq_len, config.intermediate_size}, LLAISYS_DTYPE_F32);

    ops::linear(gate, ffn_norm_hidden, gate_proj_weight, no_bias);
    ops::linear(up, ffn_norm_hidden, up_proj_weight, no_bias);

    auto swiglu_out = createTensor({seq_len, config.intermediate_size}, LLAISYS_DTYPE_F32);
    ops::swiglu(swiglu_out, gate, up);

    auto down_proj_weight = getWeight("model.layers." + std::to_string(layer_idx) + ".mlp.down_proj.weight");
    auto ffn_out = createTensor({seq_len, config.hidden_size}, LLAISYS_DTYPE_F32);
    ops::linear(ffn_out, swiglu_out, down_proj_weight, no_bias);

    ops::add(hidden_states, hidden_states, ffn_out);
}

void Qwen2Model::finalNormAndHead(tensor_t hidden_states, tensor_t logits) {
    size_t seq_len = hidden_states->shape()[0];
    auto last_hidden = hidden_states->slice(0, seq_len - 1, seq_len);
    auto last_hidden_2d = last_hidden->view({1, config.hidden_size});

    auto norm_weight = getWeight("model.norm.weight");
    auto normed = createTensor({1, config.hidden_size}, LLAISYS_DTYPE_F32);
    ops::rms_norm(normed, last_hidden_2d, norm_weight, config.rms_norm_eps);

    auto lm_head_weight = getWeight("lm_head.weight");
    auto no_bias = createTensor({0}, LLAISYS_DTYPE_F32);
    auto logits_2d = createTensor({1, config.vocab_size}, LLAISYS_DTYPE_F32);
    ops::linear(logits_2d, normed, lm_head_weight, no_bias);

    float *src = reinterpret_cast<float *>(logits_2d->data());
    float *dst = reinterpret_cast<float *>(logits->data());
    llaisys::device::getRuntimeAPI(device_type)
        ->memcpy_sync(dst, src, config.vocab_size * sizeof(float), LLAISYS_MEMCPY_D2D);
}

void Qwen2Model::forward(tensor_t input_ids, tensor_t output_logits) {
    try {
        size_t seq_len = input_ids->shape()[0];
        size_t start_pos = kv_caches[0].current_seq_len;

        auto hidden_states = createTensor({seq_len, config.hidden_size}, LLAISYS_DTYPE_F32);

        embedding(hidden_states, input_ids);

        for (size_t i = 0; i < config.num_hidden_layers; i++) {
            transformerLayer(hidden_states, i, start_pos, seq_len);
        }

        finalNormAndHead(hidden_states, output_logits);
    } catch (const std::exception &e) {
        fprintf(stderr, "FATAL: Qwen2Model::forward exception: %s\n", e.what());
        throw;
    } catch (...) {
        fprintf(stderr, "FATAL: Qwen2Model::forward unknown exception\n");
        throw;
    }
}

} // namespace models
} // namespace llaisys