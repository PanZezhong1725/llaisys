#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../core/llaisys_core.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../tensor/tensor.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using llaisys::tensor_t;

llaisysTensor_t create_tensor(
    const std::vector<size_t> &shape,
    llaisysDataType_t dtype,
    llaisysDeviceType_t device,
    int device_id) {
    return new LlaisysTensor{
        llaisys::Tensor::create(shape, dtype, device, device_id)};
}

void destroy_tensor(llaisysTensor_t &tensor) {
    delete tensor;
    tensor = nullptr;
}

} // namespace

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;
    LlaisysQwen2Weights weights{};
    std::vector<tensor_t> key_cache;
    std::vector<tensor_t> value_cache;
    size_t cache_capacity = 0;
    size_t past_length = 0;

    ~LlaisysQwen2Model() {
        const auto destroy_array = [&](llaisysTensor_t *&tensors) {
            if (tensors == nullptr) {
                return;
            }
            for (size_t layer = 0; layer < meta.nlayer; ++layer) {
                destroy_tensor(tensors[layer]);
            }
            delete[] tensors;
            tensors = nullptr;
        };

        destroy_tensor(weights.in_embed);
        destroy_tensor(weights.out_embed);
        destroy_tensor(weights.out_norm_w);
        destroy_array(weights.attn_norm_w);
        destroy_array(weights.attn_q_w);
        destroy_array(weights.attn_q_b);
        destroy_array(weights.attn_k_w);
        destroy_array(weights.attn_k_b);
        destroy_array(weights.attn_v_w);
        destroy_array(weights.attn_v_b);
        destroy_array(weights.attn_o_w);
        destroy_array(weights.mlp_norm_w);
        destroy_array(weights.mlp_gate_w);
        destroy_array(weights.mlp_up_w);
        destroy_array(weights.mlp_down_w);
    }

    tensor_t tensor(const std::vector<size_t> &shape, llaisysDataType_t dtype_) const {
        return llaisys::Tensor::create(shape, dtype_, device, device_id);
    }

    tensor_t tensor(const std::vector<size_t> &shape) const {
        return tensor(shape, meta.dtype);
    }

    void ensure_cache(size_t required_capacity) {
        if (required_capacity <= cache_capacity) {
            return;
        }
        if (required_capacity > meta.maxseq) {
            throw std::runtime_error("Qwen2: sequence exceeds maximum context length.");
        }

        size_t new_capacity = std::max<size_t>(1, cache_capacity);
        while (new_capacity < required_capacity) {
            new_capacity = std::min(meta.maxseq, new_capacity * 2);
        }

        llaisys::core::context().setDevice(device, device_id);
        const auto *runtime = llaisys::core::context().runtime().api();
        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            auto new_key = tensor({new_capacity, meta.nkvh, meta.dh});
            auto new_value = tensor({new_capacity, meta.nkvh, meta.dh});
            if (past_length > 0) {
                const size_t bytes = past_length * meta.nkvh * meta.dh
                                     * new_key->elementSize();
                runtime->memcpy_sync(
                    new_key->data(), key_cache[layer]->data(), bytes,
                    LLAISYS_MEMCPY_D2D);
                runtime->memcpy_sync(
                    new_value->data(), value_cache[layer]->data(), bytes,
                    LLAISYS_MEMCPY_D2D);
            }
            key_cache[layer] = std::move(new_key);
            value_cache[layer] = std::move(new_value);
        }
        cache_capacity = new_capacity;
    }
};

namespace {

void allocate_weight_arrays(LlaisysQwen2Model *model) {
    const size_t layers = model->meta.nlayer;
    model->weights.attn_norm_w = new llaisysTensor_t[layers]();
    model->weights.attn_q_w = new llaisysTensor_t[layers]();
    model->weights.attn_q_b = new llaisysTensor_t[layers]();
    model->weights.attn_k_w = new llaisysTensor_t[layers]();
    model->weights.attn_k_b = new llaisysTensor_t[layers]();
    model->weights.attn_v_w = new llaisysTensor_t[layers]();
    model->weights.attn_v_b = new llaisysTensor_t[layers]();
    model->weights.attn_o_w = new llaisysTensor_t[layers]();
    model->weights.mlp_norm_w = new llaisysTensor_t[layers]();
    model->weights.mlp_gate_w = new llaisysTensor_t[layers]();
    model->weights.mlp_up_w = new llaisysTensor_t[layers]();
    model->weights.mlp_down_w = new llaisysTensor_t[layers]();
}

void allocate_weights(LlaisysQwen2Model *model) {
    const auto dtype = model->meta.dtype;
    const auto device = model->device;
    const int device_id = model->device_id;
    const auto make = [&](const std::vector<size_t> &shape) {
        return create_tensor(shape, dtype, device, device_id);
    };

    model->weights.in_embed = make({model->meta.voc, model->meta.hs});
    model->weights.out_embed = make({model->meta.voc, model->meta.hs});
    model->weights.out_norm_w = make({model->meta.hs});
    allocate_weight_arrays(model);

    for (size_t layer = 0; layer < model->meta.nlayer; ++layer) {
        model->weights.attn_norm_w[layer] = make({model->meta.hs});
        model->weights.attn_q_w[layer] = make({model->meta.nh * model->meta.dh, model->meta.hs});
        model->weights.attn_q_b[layer] = make({model->meta.nh * model->meta.dh});
        model->weights.attn_k_w[layer] = make({model->meta.nkvh * model->meta.dh, model->meta.hs});
        model->weights.attn_k_b[layer] = make({model->meta.nkvh * model->meta.dh});
        model->weights.attn_v_w[layer] = make({model->meta.nkvh * model->meta.dh, model->meta.hs});
        model->weights.attn_v_b[layer] = make({model->meta.nkvh * model->meta.dh});
        model->weights.attn_o_w[layer] = make({model->meta.hs, model->meta.nh * model->meta.dh});
        model->weights.mlp_norm_w[layer] = make({model->meta.hs});
        model->weights.mlp_gate_w[layer] = make({model->meta.di, model->meta.hs});
        model->weights.mlp_up_w[layer] = make({model->meta.di, model->meta.hs});
        model->weights.mlp_down_w[layer] = make({model->meta.hs, model->meta.di});
    }
}

void validate_meta(const LlaisysQwen2Meta &meta) {
    CHECK_ARGUMENT(meta.nlayer > 0, "Qwen2: number of layers must be positive.");
    CHECK_ARGUMENT(meta.hs > 0 && meta.nh > 0 && meta.nkvh > 0 && meta.dh > 0,
                   "Qwen2: attention dimensions must be positive.");
    CHECK_ARGUMENT(meta.di > 0 && meta.maxseq > 0 && meta.voc > 0,
                   "Qwen2: model dimensions must be positive.");
    CHECK_ARGUMENT(meta.nh % meta.nkvh == 0,
                   "Qwen2: query heads must be divisible by KV heads.");
    CHECK_ARGUMENT(meta.hs == meta.nh * meta.dh,
                   "Qwen2: hidden size must equal query heads times head dimension.");
    CHECK_ARGUMENT(meta.dh % 2 == 0,
                   "Qwen2: head dimension must be even.");
    CHECK_ARGUMENT(meta.epsilon > 0.0f && meta.theta > 0.0f,
                   "Qwen2: epsilon and RoPE theta must be positive.");
    CHECK_ARGUMENT(meta.dtype == LLAISYS_DTYPE_F32
                       || meta.dtype == LLAISYS_DTYPE_F16
                       || meta.dtype == LLAISYS_DTYPE_BF16,
                   "Qwen2: model dtype must be float32, float16, or bfloat16.");
}

} // namespace

__C {
LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) try {
    CHECK_ARGUMENT(meta != nullptr, "Qwen2: metadata cannot be null.");
    CHECK_ARGUMENT(ndevice == 1, "Qwen2: exactly one device is currently supported.");
    CHECK_ARGUMENT(device == LLAISYS_DEVICE_CPU,
                   "Qwen2: only CPU inference is currently supported.");
    validate_meta(*meta);

    auto *model = new LlaisysQwen2Model();
    try {
        model->meta = *meta;
        model->device = device;
        model->device_id = device_ids == nullptr ? 0 : device_ids[0];
        model->key_cache.resize(meta->nlayer);
        model->value_cache.resize(meta->nlayer);
        allocate_weights(model);
    } catch (...) {
        delete model;
        return nullptr;
    }
    return model;
} catch (...) {
    return nullptr;
}

void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
    delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
    return model == nullptr ? nullptr : &model->weights;
}

void llaisysQwen2ModelReset(LlaisysQwen2Model *model) {
    if (model != nullptr) {
        model->past_length = 0;
    }
}

int64_t llaisysQwen2ModelInfer(
    LlaisysQwen2Model *model,
    const int64_t *token_ids,
    size_t ntoken) try {
    CHECK_ARGUMENT(model != nullptr, "Qwen2: model cannot be null.");
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen2: token IDs cannot be null.");
    CHECK_ARGUMENT(ntoken > 0, "Qwen2: at least one input token is required.");
    CHECK_ARGUMENT(model->past_length + ntoken <= model->meta.maxseq,
                   "Qwen2: sequence exceeds maximum context length.");

    model->ensure_cache(model->past_length + ntoken);
    const auto tensor = [&](const std::vector<size_t> &shape) {
        return model->tensor(shape);
    };

    auto indices = model->tensor({ntoken}, LLAISYS_DTYPE_I64);
    indices->load(token_ids);
    auto hidden = tensor({ntoken, model->meta.hs});
    llaisys::ops::embedding(hidden, indices, model->weights.in_embed->tensor);

    std::vector<int64_t> positions(ntoken);
    for (size_t token = 0; token < ntoken; ++token) {
        positions[token] = static_cast<int64_t>(model->past_length + token);
    }
    auto position_ids = model->tensor({ntoken}, LLAISYS_DTYPE_I64);
    position_ids->load(positions.data());

    llaisys::core::context().setDevice(model->device, model->device_id);
    const auto *runtime = llaisys::core::context().runtime().api();
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(model->meta.dh));

    for (size_t layer = 0; layer < model->meta.nlayer; ++layer) {
        auto normalized = tensor({ntoken, model->meta.hs});
        llaisys::ops::rms_norm(
            normalized, hidden, model->weights.attn_norm_w[layer]->tensor,
            model->meta.epsilon);

        auto query_flat = tensor({ntoken, model->meta.nh * model->meta.dh});
        auto key_flat = tensor({ntoken, model->meta.nkvh * model->meta.dh});
        auto value_flat = tensor({ntoken, model->meta.nkvh * model->meta.dh});
        llaisys::ops::linear(
            query_flat, normalized, model->weights.attn_q_w[layer]->tensor,
            model->weights.attn_q_b[layer]->tensor);
        llaisys::ops::linear(
            key_flat, normalized, model->weights.attn_k_w[layer]->tensor,
            model->weights.attn_k_b[layer]->tensor);
        llaisys::ops::linear(
            value_flat, normalized, model->weights.attn_v_w[layer]->tensor,
            model->weights.attn_v_b[layer]->tensor);

        auto query = query_flat->view({ntoken, model->meta.nh, model->meta.dh});
        auto key = key_flat->view({ntoken, model->meta.nkvh, model->meta.dh});
        auto value = value_flat->view({ntoken, model->meta.nkvh, model->meta.dh});
        auto rotated_query = tensor({ntoken, model->meta.nh, model->meta.dh});
        auto rotated_key = tensor({ntoken, model->meta.nkvh, model->meta.dh});
        llaisys::ops::rope(
            rotated_query, query, position_ids, model->meta.theta);
        llaisys::ops::rope(
            rotated_key, key, position_ids, model->meta.theta);

        const size_t cache_token_size = model->meta.nkvh * model->meta.dh
                                        * rotated_key->elementSize();
        runtime->memcpy_sync(
            model->key_cache[layer]->data() + model->past_length * cache_token_size,
            rotated_key->data(), ntoken * cache_token_size,
            LLAISYS_MEMCPY_D2D);
        runtime->memcpy_sync(
            model->value_cache[layer]->data() + model->past_length * cache_token_size,
            value->data(), ntoken * cache_token_size,
            LLAISYS_MEMCPY_D2D);

        const size_t total_length = model->past_length + ntoken;
        auto cached_key = model->key_cache[layer]->slice(0, 0, total_length);
        auto cached_value = model->value_cache[layer]->slice(0, 0, total_length);
        auto attention = tensor({ntoken, model->meta.nh, model->meta.dh});
        llaisys::ops::self_attention(
            attention, rotated_query, cached_key, cached_value, attention_scale);

        auto attention_flat = attention->view({ntoken, model->meta.hs});
        auto attention_output = tensor({ntoken, model->meta.hs});
        llaisys::ops::linear(
            attention_output, attention_flat,
            model->weights.attn_o_w[layer]->tensor, nullptr);
        auto attention_residual = tensor({ntoken, model->meta.hs});
        llaisys::ops::add(attention_residual, hidden, attention_output);

        auto mlp_normalized = tensor({ntoken, model->meta.hs});
        llaisys::ops::rms_norm(
            mlp_normalized, attention_residual,
            model->weights.mlp_norm_w[layer]->tensor, model->meta.epsilon);
        auto gate = tensor({ntoken, model->meta.di});
        auto up = tensor({ntoken, model->meta.di});
        llaisys::ops::linear(
            gate, mlp_normalized, model->weights.mlp_gate_w[layer]->tensor, nullptr);
        llaisys::ops::linear(
            up, mlp_normalized, model->weights.mlp_up_w[layer]->tensor, nullptr);
        auto activated = tensor({ntoken, model->meta.di});
        llaisys::ops::swiglu(activated, gate, up);
        auto down = tensor({ntoken, model->meta.hs});
        llaisys::ops::linear(
            down, activated, model->weights.mlp_down_w[layer]->tensor, nullptr);
        auto next_hidden = tensor({ntoken, model->meta.hs});
        llaisys::ops::add(next_hidden, attention_residual, down);
        hidden = std::move(next_hidden);
    }

    auto final_normalized = tensor({ntoken, model->meta.hs});
    llaisys::ops::rms_norm(
        final_normalized, hidden, model->weights.out_norm_w->tensor,
        model->meta.epsilon);
    auto last_hidden = final_normalized->slice(0, ntoken - 1, ntoken);
    auto logits = tensor({1, model->meta.voc});
    llaisys::ops::linear(
        logits, last_hidden, model->weights.out_embed->tensor, nullptr);

    auto max_index = model->tensor({1}, LLAISYS_DTYPE_I64);
    auto max_value = tensor({1});
    llaisys::ops::argmax(max_index, max_value, logits->view({model->meta.voc}));
    int64_t next_token = 0;
    runtime->memcpy_sync(
        &next_token, max_index->data(), sizeof(next_token), LLAISYS_MEMCPY_D2H);
    model->past_length += ntoken;
    return next_token;
} catch (...) {
    return -1;
}
}
