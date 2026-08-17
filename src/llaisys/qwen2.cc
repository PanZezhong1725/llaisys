#include "llaisys/models/qwen2.h"

#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/add/op.hpp"
#include "../tensor/tensor.hpp"
#include "../utils.hpp"

#include <cstring>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;
    std::unordered_map<std::string, llaisys::tensor_t> weights;
    std::vector<llaisys::tensor_t> key_cache;
    std::vector<llaisys::tensor_t> value_cache;
    size_t cache_len = 0;
};

namespace {
using namespace llaisys;

llaisys::tensor_t weight(LlaisysQwen2Model *model, const std::string &name) {
    auto it = model->weights.find(name);
    CHECK_ARGUMENT(it != model->weights.end(), "missing Qwen2 weight: " + name);
    return it->second;
}

llaisys::tensor_t make_tensor(const std::vector<size_t> &shape, llaisysDataType_t dtype, llaisysDeviceType_t device, int device_id) {
    return Tensor::create(shape, dtype, device, device_id);
}

llaisys::tensor_t layer_weight(LlaisysQwen2Model *model, size_t layer, const char *suffix) {
    return weight(model, "model.layers." + std::to_string(layer) + "." + suffix + ".weight");
}

llaisys::tensor_t layer_bias(LlaisysQwen2Model *model, size_t layer, const char *suffix) {
    auto it = model->weights.find("model.layers." + std::to_string(layer) + "." + suffix + ".bias");
    return it == model->weights.end() ? nullptr : it->second;
}

} // namespace

extern "C" {
LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
    CHECK_ARGUMENT(meta != nullptr, "Qwen2 metadata must not be null");
    CHECK_ARGUMENT(ndevice > 0 && device_ids != nullptr, "Qwen2 requires a device");
    auto *model = new LlaisysQwen2Model;
    model->meta = *meta;
    model->device = device;
    model->device_id = device_ids[0];
    llaisys::core::context().setDevice(device, model->device_id);
    return model;
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) { delete model; }

void llaisysQwen2ModelResetCache(LlaisysQwen2Model *model) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
    model->key_cache.clear();
    model->value_cache.clear();
    model->cache_len = 0;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *) { return nullptr; }

void llaisysQwen2ModelLoadWeight(LlaisysQwen2Model *model, const char *name, const size_t *shape, size_t ndim, llaisysDataType_t dtype, const void *data) {
    CHECK_ARGUMENT(model != nullptr && name != nullptr && shape != nullptr && data != nullptr, "invalid Qwen2 weight arguments");
    std::vector<size_t> dims(shape, shape + ndim);
    auto tensor = llaisys::Tensor::create(dims, dtype, model->device, model->device_id);
    tensor->load(data);
    model->weights[name] = std::move(tensor);
}

int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(model != nullptr && token_ids != nullptr && ntoken > 0, "invalid Qwen2 inference input");
    const auto &m = model->meta;
    size_t previous_len = model->cache_len;
    size_t total_len = previous_len + ntoken;
    auto input = make_tensor({ntoken}, LLAISYS_DTYPE_I64, model->device, model->device_id);
    input->load(token_ids);
    auto hidden = make_tensor({ntoken, m.hs}, m.dtype, model->device, model->device_id);
    ops::embedding(hidden, input, weight(model, "model.embed_tokens.weight"));

    auto positions = make_tensor({ntoken}, LLAISYS_DTYPE_I64, model->device, model->device_id);
    std::vector<int64_t> pos(ntoken);
    for (size_t i = 0; i < ntoken; i++) pos[i] = static_cast<int64_t>(previous_len + i);
    positions->load(pos.data());

    for (size_t layer = 0; layer < m.nlayer; layer++) {
        auto residual = hidden;
        auto norm = make_tensor({ntoken, m.hs}, m.dtype, model->device, model->device_id);
        ops::rms_norm(norm, hidden, layer_weight(model, layer, "input_layernorm"), m.epsilon);

        auto q_linear = make_tensor({ntoken, m.nh * m.dh}, m.dtype, model->device, model->device_id);
        auto k_linear = make_tensor({ntoken, m.nkvh * m.dh}, m.dtype, model->device, model->device_id);
        auto v_linear = make_tensor({ntoken, m.nkvh * m.dh}, m.dtype, model->device, model->device_id);
        auto q_weight = layer_weight(model, layer, "self_attn.q_proj");
        auto q_bias = layer_bias(model, layer, "self_attn.q_proj");
        auto k_bias = layer_bias(model, layer, "self_attn.k_proj");
        auto v_bias = layer_bias(model, layer, "self_attn.v_proj");
        ops::linear(q_linear, norm, q_weight, q_bias);
        ops::linear(k_linear, norm, layer_weight(model, layer, "self_attn.k_proj"), k_bias);
        ops::linear(v_linear, norm, layer_weight(model, layer, "self_attn.v_proj"), v_bias);

        auto q = q_linear->view({ntoken, m.nh, m.dh});
        auto k = k_linear->view({ntoken, m.nkvh, m.dh});
        auto v = v_linear->view({ntoken, m.nkvh, m.dh});
        auto q_rot = make_tensor(q->shape(), m.dtype, model->device, model->device_id);
        auto k_rot = make_tensor(k->shape(), m.dtype, model->device, model->device_id);
        ops::rope(q_rot, q, positions, m.theta);
        ops::rope(k_rot, k, positions, m.theta);
        llaisys::tensor_t attention_k = k_rot;
        llaisys::tensor_t attention_v = v;
        if (previous_len > 0) {
            CHECK_ARGUMENT(model->key_cache.size() == m.nlayer && model->value_cache.size() == m.nlayer, "invalid Qwen2 KV cache");
            attention_k = make_tensor({total_len, m.nkvh, m.dh}, m.dtype, model->device, model->device_id);
            attention_v = make_tensor({total_len, m.nkvh, m.dh}, m.dtype, model->device, model->device_id);
            llaisys::core::context().setDevice(model->device, model->device_id);
            size_t old_bytes = previous_len * m.nkvh * m.dh * k_rot->elementSize();
            size_t new_bytes = ntoken * m.nkvh * m.dh * k_rot->elementSize();
            llaisys::core::context().runtime().api()->memcpy_sync(attention_k->data(), model->key_cache[layer]->data(), old_bytes, LLAISYS_MEMCPY_D2D);
            llaisys::core::context().runtime().api()->memcpy_sync(attention_k->data() + old_bytes, k_rot->data(), new_bytes, LLAISYS_MEMCPY_D2D);
            llaisys::core::context().runtime().api()->memcpy_sync(attention_v->data(), model->value_cache[layer]->data(), old_bytes, LLAISYS_MEMCPY_D2D);
            llaisys::core::context().runtime().api()->memcpy_sync(attention_v->data() + old_bytes, v->data(), new_bytes, LLAISYS_MEMCPY_D2D);
        }
        if (model->key_cache.size() <= layer) {
            model->key_cache.resize(m.nlayer);
            model->value_cache.resize(m.nlayer);
        }
        model->key_cache[layer] = attention_k;
        model->value_cache[layer] = attention_v;
        auto attn = make_tensor({ntoken, m.nh, m.dh}, m.dtype, model->device, model->device_id);
        ops::self_attention(attn, q_rot, attention_k, attention_v, 1.0f / std::sqrt(static_cast<float>(m.dh)));

        auto attn_2d = attn->view({ntoken, m.hs});
        auto projected = make_tensor({ntoken, m.hs}, m.dtype, model->device, model->device_id);
        ops::linear(projected, attn_2d, layer_weight(model, layer, "self_attn.o_proj"), layer_bias(model, layer, "self_attn.o_proj"));
        hidden = make_tensor({ntoken, m.hs}, m.dtype, model->device, model->device_id);
        ops::add(hidden, residual, projected);

        residual = hidden;
        norm = make_tensor({ntoken, m.hs}, m.dtype, model->device, model->device_id);
        ops::rms_norm(norm, hidden, layer_weight(model, layer, "post_attention_layernorm"), m.epsilon);
        auto gate = make_tensor({ntoken, m.di}, m.dtype, model->device, model->device_id);
        auto up = make_tensor({ntoken, m.di}, m.dtype, model->device, model->device_id);
        ops::linear(gate, norm, layer_weight(model, layer, "mlp.gate_proj"), nullptr);
        ops::linear(up, norm, layer_weight(model, layer, "mlp.up_proj"), nullptr);
        auto activated = make_tensor({ntoken, m.di}, m.dtype, model->device, model->device_id);
        ops::swiglu(activated, gate, up);
        auto down = make_tensor({ntoken, m.hs}, m.dtype, model->device, model->device_id);
        ops::linear(down, activated, layer_weight(model, layer, "mlp.down_proj"), nullptr);
        hidden = make_tensor({ntoken, m.hs}, m.dtype, model->device, model->device_id);
        ops::add(hidden, residual, down);
    }

    auto norm = make_tensor({ntoken, m.hs}, m.dtype, model->device, model->device_id);
    ops::rms_norm(norm, hidden, weight(model, "model.norm.weight"), m.epsilon);
    auto last = norm->slice(0, ntoken - 1, ntoken);
    auto logits = make_tensor({1, m.voc}, m.dtype, model->device, model->device_id);
    ops::linear(logits, last, weight(model, "lm_head.weight"), nullptr);
    auto idx = make_tensor({1}, LLAISYS_DTYPE_I64, model->device, model->device_id);
    auto val = make_tensor({1}, m.dtype, model->device, model->device_id);
    ops::argmax(idx, val, logits->view({m.voc}));
    int64_t result = 0;
    llaisys::core::context().setDevice(model->device, model->device_id);
    llaisys::core::context().runtime().api()->memcpy_sync(&result, idx->data(), sizeof(result), LLAISYS_MEMCPY_D2H);
    model->cache_len = total_len;
    return result;
}
}
