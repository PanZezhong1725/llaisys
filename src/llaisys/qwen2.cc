#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

llaisys::tensor_t as_tensor(llaisysTensor_t handle) {
    if (handle == nullptr) {
        return nullptr;
    }
    return handle->tensor;
}

llaisysTensor_t make_handle(const llaisys::tensor_t &tensor) {
    return new LlaisysTensor{tensor};
}

void destroy_handle(llaisysTensor_t &handle) {
    delete handle;
    handle = nullptr;
}

void copy_bytes(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    llaisys::core::context().runtime().api()->memcpy_sync(dst, src, size, kind);
}

llaisys::tensor_t require_tensor(llaisysTensor_t handle, const char *name) {
    auto tensor = as_tensor(handle);
    if (tensor == nullptr) {
        throw std::runtime_error(std::string("Missing Qwen2 weight: ") + name);
    }
    return tensor;
}

} // namespace

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;
    LlaisysQwen2Weights weights{};
    std::vector<llaisysTensor_t> key_cache;
    std::vector<llaisysTensor_t> value_cache;
    size_t cache_length = 0;
    size_t cache_capacity = 0;

    void reset_cache() {
        cache_length = 0;
    }

    void ensure_cache(size_t required) {
        if (required <= cache_capacity) {
            return;
        }
        size_t new_capacity = std::max<size_t>(256, cache_capacity);
        while (new_capacity < required) {
            new_capacity = std::min(meta.maxseq, new_capacity * 2);
        }
        if (key_cache.empty()) {
            key_cache.resize(meta.nlayer, nullptr);
            value_cache.resize(meta.nlayer, nullptr);
        }
        for (size_t i = 0; i < meta.nlayer; ++i) {
            auto new_key = llaisys::Tensor::create({new_capacity, meta.nkvh, meta.dh}, meta.dtype, device, device_id);
            auto new_value = llaisys::Tensor::create({new_capacity, meta.nkvh, meta.dh}, meta.dtype, device, device_id);
            if (cache_length > 0) {
                const size_t bytes = cache_length * meta.nkvh * meta.dh * new_key->elementSize();
                copy_bytes(new_key->data(), as_tensor(key_cache[i])->data(), bytes, LLAISYS_MEMCPY_D2D);
                copy_bytes(new_value->data(), as_tensor(value_cache[i])->data(), bytes, LLAISYS_MEMCPY_D2D);
            }
            destroy_handle(key_cache[i]);
            destroy_handle(value_cache[i]);
            key_cache[i] = make_handle(new_key);
            value_cache[i] = make_handle(new_value);
        }
        cache_capacity = new_capacity;
    }

    int64_t infer(const int64_t *token_ids, size_t ntoken) {
        if (token_ids == nullptr || ntoken == 0) {
            throw std::invalid_argument("Qwen2 inference requires at least one token");
        }
        if (ntoken > 1) {
            reset_cache();
        }
        if (cache_length + ntoken > meta.maxseq) {
            throw std::invalid_argument("Qwen2 sequence exceeds maxseq");
        }
        ensure_cache(cache_length + ntoken);

        auto input_ids = llaisys::Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device, device_id);
        input_ids->load(token_ids);
        auto hidden = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, device, device_id);
        llaisys::ops::embedding(hidden, input_ids, require_tensor(weights.in_embed, "model.embed_tokens.weight"));

        auto positions = llaisys::Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device, device_id);
        std::vector<int64_t> position_values(ntoken);
        for (size_t i = 0; i < ntoken; ++i) {
            position_values[i] = static_cast<int64_t>(cache_length + i);
        }
        positions->load(position_values.data());

        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            auto normed = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, device, device_id);
            llaisys::ops::rms_norm(normed, hidden, require_tensor(weights.attn_norm_w[layer], "input_layernorm"),
                                   meta.epsilon);

            auto q_flat = llaisys::Tensor::create({ntoken, meta.nh * meta.dh}, meta.dtype, device, device_id);
            auto k_flat = llaisys::Tensor::create({ntoken, meta.nkvh * meta.dh}, meta.dtype, device, device_id);
            auto v_flat = llaisys::Tensor::create({ntoken, meta.nkvh * meta.dh}, meta.dtype, device, device_id);
            llaisys::ops::linear(q_flat, normed, require_tensor(weights.attn_q_w[layer], "q_proj"),
                                 as_tensor(weights.attn_q_b[layer]));
            llaisys::ops::linear(k_flat, normed, require_tensor(weights.attn_k_w[layer], "k_proj"),
                                 as_tensor(weights.attn_k_b[layer]));
            llaisys::ops::linear(v_flat, normed, require_tensor(weights.attn_v_w[layer], "v_proj"),
                                 as_tensor(weights.attn_v_b[layer]));

            auto q = q_flat->view({ntoken, meta.nh, meta.dh});
            auto k = k_flat->view({ntoken, meta.nkvh, meta.dh});
            auto v = v_flat->view({ntoken, meta.nkvh, meta.dh});
            auto q_rot = llaisys::Tensor::create({ntoken, meta.nh, meta.dh}, meta.dtype, device, device_id);
            auto k_rot = llaisys::Tensor::create({ntoken, meta.nkvh, meta.dh}, meta.dtype, device, device_id);
            llaisys::ops::rope(q_rot, q, positions, meta.theta);
            llaisys::ops::rope(k_rot, k, positions, meta.theta);

            auto key_dst = as_tensor(key_cache[layer])->slice(0, cache_length, cache_length + ntoken);
            auto value_dst = as_tensor(value_cache[layer])->slice(0, cache_length, cache_length + ntoken);
            copy_bytes(key_dst->data(), k_rot->data(), k_rot->numel() * k_rot->elementSize(), LLAISYS_MEMCPY_D2D);
            copy_bytes(value_dst->data(), v->data(), v->numel() * v->elementSize(), LLAISYS_MEMCPY_D2D);
            auto keys = as_tensor(key_cache[layer])->slice(0, 0, cache_length + ntoken);
            auto values = as_tensor(value_cache[layer])->slice(0, 0, cache_length + ntoken);

            auto attention = llaisys::Tensor::create({ntoken, meta.nh, meta.dh}, meta.dtype, device, device_id);
            llaisys::ops::self_attention(attention, q_rot, keys, values,
                                         1.0F / std::sqrt(static_cast<float>(meta.dh)));
            auto attention_flat = attention->view({ntoken, meta.hs});
            auto projected = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, device, device_id);
            llaisys::ops::linear(projected, attention_flat, require_tensor(weights.attn_o_w[layer], "o_proj"), nullptr);
            auto residual = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, device, device_id);
            llaisys::ops::add(residual, hidden, projected);

            auto mlp_normed = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, device, device_id);
            llaisys::ops::rms_norm(mlp_normed, residual, require_tensor(weights.mlp_norm_w[layer], "post_attention_layernorm"),
                                   meta.epsilon);
            auto gate = llaisys::Tensor::create({ntoken, meta.di}, meta.dtype, device, device_id);
            auto up = llaisys::Tensor::create({ntoken, meta.di}, meta.dtype, device, device_id);
            auto mlp = llaisys::Tensor::create({ntoken, meta.di}, meta.dtype, device, device_id);
            llaisys::ops::linear(gate, mlp_normed, require_tensor(weights.mlp_gate_w[layer], "gate_proj"), nullptr);
            llaisys::ops::linear(up, mlp_normed, require_tensor(weights.mlp_up_w[layer], "up_proj"), nullptr);
            llaisys::ops::swiglu(mlp, gate, up);
            auto down = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, device, device_id);
            llaisys::ops::linear(down, mlp, require_tensor(weights.mlp_down_w[layer], "down_proj"), nullptr);
            hidden = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, device, device_id);
            llaisys::ops::add(hidden, residual, down);
        }

        cache_length += ntoken;
        auto final_norm = llaisys::Tensor::create({ntoken, meta.hs}, meta.dtype, device, device_id);
        llaisys::ops::rms_norm(final_norm, hidden, require_tensor(weights.out_norm_w, "model.norm.weight"), meta.epsilon);
        auto logits = llaisys::Tensor::create({ntoken, meta.voc}, meta.dtype, device, device_id);
        auto output_weight = weights.out_embed != nullptr ? weights.out_embed : weights.in_embed;
        llaisys::ops::linear(logits, final_norm, require_tensor(output_weight, "lm_head.weight"), nullptr);
        auto last_logits = logits->slice(0, ntoken - 1, ntoken)->view({meta.voc});
        auto max_index = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
        auto max_value = llaisys::Tensor::create({1}, meta.dtype, device, device_id);
        llaisys::ops::argmax(max_index, max_value, last_logits);
        int64_t result = 0;
        copy_bytes(&result, max_index->data(), sizeof(result), LLAISYS_MEMCPY_D2H);
        return result;
    }
};

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                               llaisysDeviceType_t device,
                                               int *device_ids,
                                               int ndevice) {
        if (meta == nullptr || ndevice <= 0 || device_ids == nullptr) {
            return nullptr;
        }
        auto *model = new LlaisysQwen2Model;
        model->meta = *meta;
        model->device = device;
        model->device_id = device_ids[0];
        model->weights.attn_norm_w = new llaisysTensor_t[meta->nlayer]();
        model->weights.attn_q_w = new llaisysTensor_t[meta->nlayer]();
        model->weights.attn_q_b = new llaisysTensor_t[meta->nlayer]();
        model->weights.attn_k_w = new llaisysTensor_t[meta->nlayer]();
        model->weights.attn_k_b = new llaisysTensor_t[meta->nlayer]();
        model->weights.attn_v_w = new llaisysTensor_t[meta->nlayer]();
        model->weights.attn_v_b = new llaisysTensor_t[meta->nlayer]();
        model->weights.attn_o_w = new llaisysTensor_t[meta->nlayer]();
        model->weights.mlp_norm_w = new llaisysTensor_t[meta->nlayer]();
        model->weights.mlp_gate_w = new llaisysTensor_t[meta->nlayer]();
        model->weights.mlp_up_w = new llaisysTensor_t[meta->nlayer]();
        model->weights.mlp_down_w = new llaisysTensor_t[meta->nlayer]();
        return model;
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
        if (model == nullptr) return;
        destroy_handle(model->weights.in_embed);
        destroy_handle(model->weights.out_embed);
        destroy_handle(model->weights.out_norm_w);
        for (auto &handle : model->key_cache) destroy_handle(handle);
        for (auto &handle : model->value_cache) destroy_handle(handle);
        const size_t n = model->meta.nlayer;
        llaisysTensor_t *arrays[] = {model->weights.attn_norm_w, model->weights.attn_q_w, model->weights.attn_q_b,
            model->weights.attn_k_w, model->weights.attn_k_b, model->weights.attn_v_w, model->weights.attn_v_b,
            model->weights.attn_o_w, model->weights.mlp_norm_w, model->weights.mlp_gate_w, model->weights.mlp_up_w,
            model->weights.mlp_down_w};
        for (auto array : arrays) {
            if (array != nullptr) {
                for (size_t i = 0; i < n; ++i) destroy_handle(array[i]);
                delete[] array;
            }
        }
        delete model;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        return model == nullptr ? nullptr : &model->weights;
    }

    void llaisysQwen2ModelLoadWeight(LlaisysQwen2Model *model, const char *name, const void *data,
                                     const size_t *shape, size_t ndim, llaisysDataType_t dtype) {
        try {
        if (model == nullptr || name == nullptr || data == nullptr || shape == nullptr) {
            throw std::invalid_argument("Invalid Qwen2 weight arguments");
        }
        const std::string key(name);
        llaisysTensor_t *target = nullptr;
        size_t layer = 0;
        const auto set_layer_target = [&](llaisysTensor_t *array) { target = &array[layer]; };
        if (key == "model.embed_tokens.weight") target = &model->weights.in_embed;
        else if (key == "lm_head.weight") target = &model->weights.out_embed;
        else if (key == "model.norm.weight") target = &model->weights.out_norm_w;
        else {
            const std::string prefix = "model.layers.";
            if (key.rfind(prefix, 0) != 0) throw std::invalid_argument("Unknown Qwen2 weight: " + key);
            const size_t dot = key.find('.', prefix.size());
            if (dot == std::string::npos) throw std::invalid_argument("Malformed Qwen2 weight: " + key);
            layer = static_cast<size_t>(std::stoul(key.substr(prefix.size(), dot - prefix.size())));
            if (layer >= model->meta.nlayer) throw std::invalid_argument("Qwen2 layer out of range");
            const std::string suffix = key.substr(dot + 1);
            if (suffix == "input_layernorm.weight") set_layer_target(model->weights.attn_norm_w);
            else if (suffix == "self_attn.q_proj.weight") set_layer_target(model->weights.attn_q_w);
            else if (suffix == "self_attn.q_proj.bias") set_layer_target(model->weights.attn_q_b);
            else if (suffix == "self_attn.k_proj.weight") set_layer_target(model->weights.attn_k_w);
            else if (suffix == "self_attn.k_proj.bias") set_layer_target(model->weights.attn_k_b);
            else if (suffix == "self_attn.v_proj.weight") set_layer_target(model->weights.attn_v_w);
            else if (suffix == "self_attn.v_proj.bias") set_layer_target(model->weights.attn_v_b);
            else if (suffix == "self_attn.o_proj.weight") set_layer_target(model->weights.attn_o_w);
            else if (suffix == "post_attention_layernorm.weight") set_layer_target(model->weights.mlp_norm_w);
            else if (suffix == "mlp.gate_proj.weight") set_layer_target(model->weights.mlp_gate_w);
            else if (suffix == "mlp.up_proj.weight") set_layer_target(model->weights.mlp_up_w);
            else if (suffix == "mlp.down_proj.weight") set_layer_target(model->weights.mlp_down_w);
            else throw std::invalid_argument("Unknown Qwen2 weight: " + key);
        }
        if (*target != nullptr) destroy_handle(*target);
        std::vector<size_t> shape_vec(shape, shape + ndim);
        auto tensor = llaisys::Tensor::create(shape_vec, dtype, model->device, model->device_id);
        tensor->load(data);
        *target = make_handle(tensor);
        } catch (...) {
            std::terminate();
        }
    }

    void llaisysQwen2ModelReset(LlaisysQwen2Model *model) {
        if (model != nullptr) model->reset_cache();
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        try {
            if (model == nullptr) throw std::invalid_argument("Qwen2 model is null");
            return model->infer(token_ids, ntoken);
        } catch (...) {
            std::terminate();
        }
    }
}
