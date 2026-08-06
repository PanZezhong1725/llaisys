#include "qwen2.hpp"

#include "../core/llaisys_core.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils.hpp"

#include <cmath>
#include <cstring>
#include <numeric>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const Qwen2Meta &meta, llaisysDeviceType_t device_type, int device_id)
    : _meta(meta), _device_type(device_type), _device_id(device_id), _cache_len(0) {
    // Initialize KV cache
    _k_cache.resize(_meta.nlayer);
    _v_cache.resize(_meta.nlayer);
    
    for (size_t i = 0; i < _meta.nlayer; i++) {
        _k_cache[i] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        _v_cache[i] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
    }
    
    // Initialize weights structure
    _weights.attn_norm_w.resize(_meta.nlayer);
    _weights.attn_q_w.resize(_meta.nlayer);
    _weights.attn_q_b.resize(_meta.nlayer);
    _weights.attn_k_w.resize(_meta.nlayer);
    _weights.attn_k_b.resize(_meta.nlayer);
    _weights.attn_v_w.resize(_meta.nlayer);
    _weights.attn_v_b.resize(_meta.nlayer);
    _weights.attn_o_w.resize(_meta.nlayer);
    _weights.mlp_norm_w.resize(_meta.nlayer);
    _weights.mlp_gate_w.resize(_meta.nlayer);
    _weights.mlp_up_w.resize(_meta.nlayer);
    _weights.mlp_down_w.resize(_meta.nlayer);
    
    // Create weight tensors
    _weights.in_embed = Tensor::create({_meta.voc, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _weights.out_embed = Tensor::create({_meta.voc, _meta.hs}, _meta.dtype, _device_type, _device_id);
    _weights.out_norm_w = Tensor::create({_meta.hs}, _meta.dtype, _device_type, _device_id);
    
    for (size_t i = 0; i < _meta.nlayer; i++) {
        _weights.attn_norm_w[i] = Tensor::create({_meta.hs}, _meta.dtype, _device_type, _device_id);
        _weights.attn_q_w[i] = Tensor::create({_meta.nh * _meta.dh, _meta.hs}, _meta.dtype, _device_type, _device_id);
        _weights.attn_q_b[i] = Tensor::create({_meta.nh * _meta.dh}, _meta.dtype, _device_type, _device_id);
        _weights.attn_k_w[i] = Tensor::create({_meta.nkvh * _meta.dh, _meta.hs}, _meta.dtype, _device_type, _device_id);
        _weights.attn_k_b[i] = Tensor::create({_meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);
        _weights.attn_v_w[i] = Tensor::create({_meta.nkvh * _meta.dh, _meta.hs}, _meta.dtype, _device_type, _device_id);
        _weights.attn_v_b[i] = Tensor::create({_meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);
        _weights.attn_o_w[i] = Tensor::create({_meta.hs, _meta.nh * _meta.dh}, _meta.dtype, _device_type, _device_id);
        _weights.mlp_norm_w[i] = Tensor::create({_meta.hs}, _meta.dtype, _device_type, _device_id);
        _weights.mlp_gate_w[i] = Tensor::create({_meta.di, _meta.hs}, _meta.dtype, _device_type, _device_id);
        _weights.mlp_up_w[i] = Tensor::create({_meta.di, _meta.hs}, _meta.dtype, _device_type, _device_id);
        _weights.mlp_down_w[i] = Tensor::create({_meta.hs, _meta.di}, _meta.dtype, _device_type, _device_id);
    }
}

Qwen2Weights &Qwen2Model::weights() {
    return _weights;
}

tensor_t Qwen2Model::rms_norm(tensor_t x, tensor_t weight, float eps) {
    auto out = Tensor::create(x->shape(), x->dtype(), x->deviceType(), x->deviceId());
    ops::rms_norm(out, x, weight, eps);
    return out;
}

tensor_t Qwen2Model::embedding(const std::vector<int64_t> &token_ids) {
    size_t ntoken = token_ids.size();
    auto index = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    index->load(token_ids.data());
    
    auto out = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::embedding(out, index, _weights.in_embed);
    return out;
}

tensor_t Qwen2Model::attention(size_t layer_idx, tensor_t hidden, size_t pos_offset) {
    size_t seqlen = hidden->shape()[0];
    size_t total_len = pos_offset + seqlen;
    
    // Layer norm
    auto normed = rms_norm(hidden, _weights.attn_norm_w[layer_idx], _meta.epsilon);
    
    // QKV projections
    auto q = Tensor::create({seqlen, _meta.nh * _meta.dh}, _meta.dtype, _device_type, _device_id);
    auto k = Tensor::create({seqlen, _meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);
    auto v = Tensor::create({seqlen, _meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);
    
    ops::linear(q, normed, _weights.attn_q_w[layer_idx], _weights.attn_q_b[layer_idx]);
    ops::linear(k, normed, _weights.attn_k_w[layer_idx], _weights.attn_k_b[layer_idx]);
    ops::linear(v, normed, _weights.attn_v_w[layer_idx], _weights.attn_v_b[layer_idx]);
    
    // Reshape for multi-head attention
    auto q_reshaped = q->view({seqlen, _meta.nh, _meta.dh});
    auto k_reshaped = k->view({seqlen, _meta.nkvh, _meta.dh});
    auto v_reshaped = v->view({seqlen, _meta.nkvh, _meta.dh});
    
    // Apply RoPE
    auto pos_ids = Tensor::create({seqlen}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    std::vector<int64_t> pos_data(seqlen);
    for (size_t i = 0; i < seqlen; i++) {
        pos_data[i] = static_cast<int64_t>(pos_offset + i);
    }
    pos_ids->load(pos_data.data());
    
    auto q_rope = Tensor::create({seqlen, _meta.nh, _meta.dh}, _meta.dtype, _device_type, _device_id);
    auto k_rope = Tensor::create({seqlen, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
    
    ops::rope(q_rope, q_reshaped, pos_ids, _meta.theta);
    ops::rope(k_rope, k_reshaped, pos_ids, _meta.theta);
    
    // Store to KV cache
    auto k_cache_slice = _k_cache[layer_idx]->slice(0, pos_offset, total_len);
    auto v_cache_slice = _v_cache[layer_idx]->slice(0, pos_offset, total_len);
    ops::rearrange(k_cache_slice, k_rope);
    ops::rearrange(v_cache_slice, v_reshaped);
    
    // Get full KV cache for attention
    auto k_full = _k_cache[layer_idx]->slice(0, 0, total_len);
    auto v_full = _v_cache[layer_idx]->slice(0, 0, total_len);
    
    // Self attention
    auto attn_out = Tensor::create({seqlen, _meta.nh, _meta.dh}, _meta.dtype, _device_type, _device_id);
    float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
    ops::self_attention(attn_out, q_rope, k_full, v_full, scale);
    
    // Output projection
    auto attn_reshaped = attn_out->view({seqlen, _meta.nh * _meta.dh});
    auto out = Tensor::create({seqlen, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::linear(out, attn_reshaped, _weights.attn_o_w[layer_idx], nullptr);
    
    // Residual connection
    auto result = Tensor::create({seqlen, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::add(result, hidden, out);
    
    return result;
}

tensor_t Qwen2Model::mlp(size_t layer_idx, tensor_t hidden) {
    size_t seqlen = hidden->shape()[0];
    
    // Layer norm
    auto normed = rms_norm(hidden, _weights.mlp_norm_w[layer_idx], _meta.epsilon);
    
    // Gate and Up projections
    auto gate = Tensor::create({seqlen, _meta.di}, _meta.dtype, _device_type, _device_id);
    auto up = Tensor::create({seqlen, _meta.di}, _meta.dtype, _device_type, _device_id);
    
    ops::linear(gate, normed, _weights.mlp_gate_w[layer_idx], nullptr);
    ops::linear(up, normed, _weights.mlp_up_w[layer_idx], nullptr);
    
    // SwiGLU activation
    auto act = Tensor::create({seqlen, _meta.di}, _meta.dtype, _device_type, _device_id);
    ops::swiglu(act, gate, up);
    
    // Down projection
    auto out = Tensor::create({seqlen, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::linear(out, act, _weights.mlp_down_w[layer_idx], nullptr);
    
    // Residual connection
    auto result = Tensor::create({seqlen, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::add(result, hidden, out);
    
    return result;
}

tensor_t Qwen2Model::forward(const std::vector<int64_t> &token_ids) {
    size_t seqlen = token_ids.size();
    size_t pos_offset = _cache_len;
    
    // Embedding
    auto hidden = embedding(token_ids);
    
    // Transformer layers
    for (size_t i = 0; i < _meta.nlayer; i++) {
        hidden = attention(i, hidden, pos_offset);
        hidden = mlp(i, hidden);
    }
    
    // Final norm
    hidden = rms_norm(hidden, _weights.out_norm_w, _meta.epsilon);
    
    // Output projection (only for the last token)
    auto last_hidden = hidden->slice(0, seqlen - 1, seqlen);
    auto logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device_type, _device_id);
    ops::linear(logits, last_hidden, _weights.out_embed, nullptr);
    
    // Update cache length
    _cache_len += seqlen;
    
    return logits;
}

int64_t Qwen2Model::infer(const std::vector<int64_t> &token_ids) {
    auto logits = forward(token_ids);
    
    // Argmax sampling
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    auto max_val = Tensor::create({1}, _meta.dtype, _device_type, _device_id);
    ops::argmax(max_idx, max_val, logits);
    
    // Copy result back to host
    int64_t result;
    core::context().setDevice(_device_type, _device_id);
    core::context().runtime().api()->memcpy_sync(
        &result,
        max_idx->data(),
        sizeof(int64_t),
        LLAISYS_MEMCPY_D2H
    );
    
    return result;
}

} // namespace llaisys::models
