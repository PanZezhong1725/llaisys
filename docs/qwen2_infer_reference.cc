// Reference implementation of llaisysQwen2ModelInfer, kept here (outside
// src/, so xmake never compiles it) for the user to check their own
// from-scratch implementation against. Not part of the build.
//
// See src/llaisys/models/qwen2.cc for the live file the user is writing.

#include "llaisys/models/qwen2.h" // 声明
#include "../../src/ops/add/op.hpp"
#include "../../src/ops/argmax/op.hpp"
#include "../../src/ops/embedding/op.hpp"
#include "../../src/ops/linear/op.hpp"
#include "../../src/ops/rms_norm/op.hpp"
#include "../../src/ops/rope/op.hpp"
#include "../../src/ops/self_attention/op.hpp"
#include "../../src/ops/swiglu/op.hpp"
#include "../../src/tensor/tensor.hpp"
#include "../../src/llaisys/llaisys_tensor.hpp"

#include <cmath>
#include <cstring>

int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
    using llaisys::Tensor;
    using llaisys::tensor_t;
    namespace ops = llaisys::ops;

    const auto &meta = model->meta;
    auto device = model->device;
    int dev_id = model->device_ids.empty() ? 0 : model->device_ids[0];
    auto dtype = meta.dtype;

    size_t cur_len = model->cur_len;
    size_t total_len = cur_len + ntoken;

    // token ids -> device tensor
    tensor_t idx = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device, dev_id);
    idx->load(token_ids);

    // position ids for the newly-fed tokens: [cur_len, cur_len+1, ...]
    std::vector<int64_t> pos_host(ntoken);
    for (size_t i = 0; i < ntoken; i++) {
        pos_host[i] = static_cast<int64_t>(cur_len + i);
    }
    tensor_t pos = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device, dev_id);
    pos->load(pos_host.data());

    tensor_t x = Tensor::create({ntoken, meta.hs}, dtype, device, dev_id);
    ops::embedding(x, idx, model->weights.in_embed->tensor);

    float scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));

    for (size_t l = 0; l < meta.nlayer; l++) {
        tensor_t normed = Tensor::create({ntoken, meta.hs}, dtype, device, dev_id);
        ops::rms_norm(normed, x, model->attn_norm_w[l]->tensor, meta.epsilon);

        tensor_t q = Tensor::create({ntoken, meta.nh * meta.dh}, dtype, device, dev_id);
        ops::linear(q, normed, model->attn_q_w[l]->tensor, model->attn_q_b[l]->tensor);
        tensor_t k = Tensor::create({ntoken, meta.nkvh * meta.dh}, dtype, device, dev_id);
        ops::linear(k, normed, model->attn_k_w[l]->tensor, model->attn_k_b[l]->tensor);
        tensor_t v = Tensor::create({ntoken, meta.nkvh * meta.dh}, dtype, device, dev_id);
        ops::linear(v, normed, model->attn_v_w[l]->tensor, model->attn_v_b[l]->tensor);

        tensor_t q3 = q->view({ntoken, meta.nh, meta.dh});
        tensor_t k3 = k->view({ntoken, meta.nkvh, meta.dh});
        tensor_t v3 = v->view({ntoken, meta.nkvh, meta.dh});

        tensor_t q_rope = Tensor::create({ntoken, meta.nh, meta.dh}, dtype, device, dev_id);
        ops::rope(q_rope, q3, pos, meta.theta);
        tensor_t k_rope = Tensor::create({ntoken, meta.nkvh, meta.dh}, dtype, device, dev_id);
        ops::rope(k_rope, k3, pos, meta.theta);

        // Both sides are always fully-contiguous [ntoken, nkvh, dh] buffers
        // (a slice of the leading dim of a contiguous cache tensor stays
        // contiguous), so a flat memcpy stands in for a rearrange copy here
        // (ops::rearrange itself is still a TO_BE_IMPLEMENTED stub).
        tensor_t k_cache_slice = model->k_cache[l]->slice(0, cur_len, total_len);
        tensor_t v_cache_slice = model->v_cache[l]->slice(0, cur_len, total_len);
        std::memcpy(k_cache_slice->data(), k_rope->data(), k_rope->numel() * k_rope->elementSize());
        std::memcpy(v_cache_slice->data(), v3->data(), v3->numel() * v3->elementSize());

        tensor_t k_all = model->k_cache[l]->slice(0, 0, total_len);
        tensor_t v_all = model->v_cache[l]->slice(0, 0, total_len);

        tensor_t attn_out = Tensor::create({ntoken, meta.nh, meta.dh}, dtype, device, dev_id);
        ops::self_attention(attn_out, q_rope, k_all, v_all, scale);

        tensor_t attn_out2 = attn_out->view({ntoken, meta.nh * meta.dh});
        tensor_t o = Tensor::create({ntoken, meta.hs}, dtype, device, dev_id);
        ops::linear(o, attn_out2, model->attn_o_w[l]->tensor, nullptr);

        tensor_t x_attn = Tensor::create({ntoken, meta.hs}, dtype, device, dev_id);
        ops::add(x_attn, x, o);
        x = x_attn;

        tensor_t normed2 = Tensor::create({ntoken, meta.hs}, dtype, device, dev_id);
        ops::rms_norm(normed2, x, model->mlp_norm_w[l]->tensor, meta.epsilon);

        tensor_t gate = Tensor::create({ntoken, meta.di}, dtype, device, dev_id);
        ops::linear(gate, normed2, model->mlp_gate_w[l]->tensor, nullptr);
        tensor_t up = Tensor::create({ntoken, meta.di}, dtype, device, dev_id);
        ops::linear(up, normed2, model->mlp_up_w[l]->tensor, nullptr);
        tensor_t swiglu_out = Tensor::create({ntoken, meta.di}, dtype, device, dev_id);
        ops::swiglu(swiglu_out, gate, up);
        tensor_t down = Tensor::create({ntoken, meta.hs}, dtype, device, dev_id);
        ops::linear(down, swiglu_out, model->mlp_down_w[l]->tensor, nullptr);

        tensor_t x_mlp = Tensor::create({ntoken, meta.hs}, dtype, device, dev_id);
        ops::add(x_mlp, x, down);
        x = x_mlp;
    }

    tensor_t final_normed = Tensor::create({ntoken, meta.hs}, dtype, device, dev_id);
    ops::rms_norm(final_normed, x, model->weights.out_norm_w->tensor, meta.epsilon);

    tensor_t logits = Tensor::create({ntoken, meta.voc}, dtype, device, dev_id);
    ops::linear(logits, final_normed, model->weights.out_embed->tensor, nullptr);

    tensor_t last_logits = logits->slice(0, ntoken - 1, ntoken)->view({meta.voc});

    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device, dev_id);
    tensor_t max_val = Tensor::create({1}, dtype, device, dev_id);
    ops::argmax(max_idx, max_val, last_logits);

    int64_t next_token = 0;
    std::memcpy(&next_token, max_idx->data(), sizeof(int64_t));

    model->cur_len = total_len;
    return next_token;
}
