#include "qwen2.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::models::qwen2 {

Qwen2::Qwen2(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device(device), _device_id(device_id) {
    ASSERT(_meta.nlayer > 0, "Qwen2: nlayer must be positive.");
    ASSERT(_meta.nkvh > 0 && _meta.nh % _meta.nkvh == 0,
           "Qwen2: nh must be a multiple of nkvh.");
    ASSERT(_meta.maxseq > 0, "Qwen2: maxseq must be positive.");

    const size_t nlayer = _meta.nlayer;
    const size_t hs = _meta.hs;
    const size_t nh = _meta.nh;
    const size_t nkvh = _meta.nkvh;
    const size_t dh = _meta.dh;
    const size_t di = _meta.di;
    const size_t voc = _meta.voc;

    _weights.in_embed = create({voc, hs});
    _weights.out_embed = create({voc, hs});
    _weights.out_norm_w = create({hs});

    auto resize_all = [nlayer](std::vector<tensor_t> &v) { v.resize(nlayer); };
    resize_all(_weights.attn_norm_w);
    resize_all(_weights.attn_q_w);
    resize_all(_weights.attn_q_b);
    resize_all(_weights.attn_k_w);
    resize_all(_weights.attn_k_b);
    resize_all(_weights.attn_v_w);
    resize_all(_weights.attn_v_b);
    resize_all(_weights.attn_o_w);
    resize_all(_weights.mlp_norm_w);
    resize_all(_weights.mlp_gate_w);
    resize_all(_weights.mlp_up_w);
    resize_all(_weights.mlp_down_w);

    _k_cache.resize(nlayer);
    _v_cache.resize(nlayer);

    for (size_t i = 0; i < nlayer; ++i) {
        _weights.attn_norm_w[i] = create({hs});
        _weights.attn_q_w[i] = create({nh * dh, hs});
        _weights.attn_q_b[i] = create({nh * dh});
        _weights.attn_k_w[i] = create({nkvh * dh, hs});
        _weights.attn_k_b[i] = create({nkvh * dh});
        _weights.attn_v_w[i] = create({nkvh * dh, hs});
        _weights.attn_v_b[i] = create({nkvh * dh});
        _weights.attn_o_w[i] = create({hs, nh * dh});
        _weights.mlp_norm_w[i] = create({hs});
        _weights.mlp_gate_w[i] = create({di, hs});
        _weights.mlp_up_w[i] = create({di, hs});
        _weights.mlp_down_w[i] = create({hs, di});

        _k_cache[i] = create({_meta.maxseq, nkvh, dh});
        _v_cache[i] = create({_meta.maxseq, nkvh, dh});
    }
}

tensor_t Qwen2::create(const std::vector<size_t> &shape) {
    return Tensor::create(shape, _meta.dtype, _device, _device_id);
}

int64_t Qwen2::infer(const int64_t *token_ids, size_t ntoken) {
    ASSERT(ntoken > 0, "Qwen2: at least one token is required.");
    const size_t past = _past_len;
    const size_t total = past + ntoken;
    ASSERT(total <= _meta.maxseq, "Qwen2: sequence length exceeds maxseq.");

    const size_t hs = _meta.hs;
    const size_t nh = _meta.nh;
    const size_t nkvh = _meta.nkvh;
    const size_t dh = _meta.dh;
    const size_t di = _meta.di;
    const size_t voc = _meta.voc;
    const float scale = 1.f / std::sqrt(static_cast<float>(dh));

    core::context().setDevice(_device, _device_id);

    // Token ids and absolute positions of the new tokens.
    auto idx = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    idx->load(token_ids);

    std::vector<int64_t> pos(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        pos[i] = static_cast<int64_t>(past + i);
    }
    auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    pos_ids->load(pos.data());

    auto hidden = create({ntoken, hs});
    ops::embedding(hidden, idx, _weights.in_embed);

    auto normed = create({ntoken, hs});
    auto q = create({ntoken, nh * dh});
    auto attn_out = create({ntoken, nh, dh});
    auto proj = create({ntoken, hs});
    auto gate = create({ntoken, di});
    auto up = create({ntoken, di});
    auto act = create({ntoken, di});

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        ops::rms_norm(normed, hidden, _weights.attn_norm_w[layer], _meta.epsilon);

        // K/V for the new tokens are written straight into their cache slots, so the
        // attention below can read the whole [0, total) history without any copying.
        auto k_slot = _k_cache[layer]->slice(0, past, total);
        auto v_slot = _v_cache[layer]->slice(0, past, total);

        ops::linear(k_slot->view({ntoken, nkvh * dh}), normed,
                    _weights.attn_k_w[layer], _weights.attn_k_b[layer]);
        ops::linear(v_slot->view({ntoken, nkvh * dh}), normed,
                    _weights.attn_v_w[layer], _weights.attn_v_b[layer]);
        ops::linear(q, normed, _weights.attn_q_w[layer], _weights.attn_q_b[layer]);

        auto q3 = q->view({ntoken, nh, dh});
        ops::rope(q3, q3, pos_ids, _meta.theta);
        ops::rope(k_slot, k_slot, pos_ids, _meta.theta);

        ops::self_attention(attn_out, q3,
                            _k_cache[layer]->slice(0, 0, total),
                            _v_cache[layer]->slice(0, 0, total), scale);

        ops::linear(proj, attn_out->view({ntoken, nh * dh}), _weights.attn_o_w[layer], nullptr);
        ops::add(hidden, hidden, proj);

        ops::rms_norm(normed, hidden, _weights.mlp_norm_w[layer], _meta.epsilon);
        ops::linear(gate, normed, _weights.mlp_gate_w[layer], nullptr);
        ops::linear(up, normed, _weights.mlp_up_w[layer], nullptr);
        ops::swiglu(act, gate, up);
        ops::linear(proj, act, _weights.mlp_down_w[layer], nullptr);
        ops::add(hidden, hidden, proj);
    }

    _past_len = total;

    // Only the last position is needed to pick the next token.
    auto last = hidden->slice(0, ntoken - 1, ntoken);
    auto last_normed = create({1, hs});
    ops::rms_norm(last_normed, last, _weights.out_norm_w, _meta.epsilon);

    auto logits = create({1, voc});
    ops::linear(logits, last_normed, _weights.out_embed, nullptr);

    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    auto max_val = create({1});
    ops::argmax(max_idx, max_val, logits->view({voc}));

    int64_t next = 0;
    if (_device == LLAISYS_DEVICE_CPU) {
        std::memcpy(&next, max_idx->data(), sizeof(next));
    } else {
        core::context().runtime().api()->memcpy_sync(
            &next, max_idx->data(), sizeof(next), LLAISYS_MEMCPY_D2H);
    }
    return next;
}

} // namespace llaisys::models::qwen2
