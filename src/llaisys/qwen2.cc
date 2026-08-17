#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../models/qwen2/model.hpp"
#include "../utils.hpp"

#include <exception>
#include <iostream>
#include <memory>
#include <vector>

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::qwen2::Qwen2Model> model;
    LlaisysQwen2Weights weights{};

    std::vector<std::unique_ptr<LlaisysTensor>> tensor_handles;
    std::vector<llaisysTensor_t> attn_norm_w;
    std::vector<llaisysTensor_t> attn_q_w;
    std::vector<llaisysTensor_t> attn_q_b;
    std::vector<llaisysTensor_t> attn_k_w;
    std::vector<llaisysTensor_t> attn_k_b;
    std::vector<llaisysTensor_t> attn_v_w;
    std::vector<llaisysTensor_t> attn_v_b;
    std::vector<llaisysTensor_t> attn_o_w;
    std::vector<llaisysTensor_t> mlp_norm_w;
    std::vector<llaisysTensor_t> mlp_gate_w;
    std::vector<llaisysTensor_t> mlp_up_w;
    std::vector<llaisysTensor_t> mlp_down_w;
};

namespace {

llaisysTensor_t makeTensorHandle(
    LlaisysQwen2Model &model,
    const llaisys::tensor_t &tensor) {
    auto handle = std::make_unique<LlaisysTensor>();
    handle->tensor = tensor;

    auto *raw_handle = handle.get();
    model.tensor_handles.push_back(std::move(handle));
    return raw_handle;
}

void buildWeightHandles(LlaisysQwen2Model &model) {
    auto &weights = model.model->weights();
    const size_t nlayer = weights.layers.size();

    model.tensor_handles.reserve(3 + 12 * nlayer);
    model.weights.in_embed = makeTensorHandle(model, weights.in_embed);
    model.weights.out_embed = makeTensorHandle(model, weights.out_embed);
    model.weights.out_norm_w = makeTensorHandle(model, weights.out_norm_w);

    model.attn_norm_w.resize(nlayer);
    model.attn_q_w.resize(nlayer);
    model.attn_q_b.resize(nlayer);
    model.attn_k_w.resize(nlayer);
    model.attn_k_b.resize(nlayer);
    model.attn_v_w.resize(nlayer);
    model.attn_v_b.resize(nlayer);
    model.attn_o_w.resize(nlayer);
    model.mlp_norm_w.resize(nlayer);
    model.mlp_gate_w.resize(nlayer);
    model.mlp_up_w.resize(nlayer);
    model.mlp_down_w.resize(nlayer);

    for (size_t i = 0; i < nlayer; ++i) {
        auto &layer = weights.layers[i];
        model.attn_norm_w[i] = makeTensorHandle(model, layer.attn_norm_w);
        model.attn_q_w[i] = makeTensorHandle(model, layer.attn_q_w);
        model.attn_q_b[i] = makeTensorHandle(model, layer.attn_q_b);
        model.attn_k_w[i] = makeTensorHandle(model, layer.attn_k_w);
        model.attn_k_b[i] = makeTensorHandle(model, layer.attn_k_b);
        model.attn_v_w[i] = makeTensorHandle(model, layer.attn_v_w);
        model.attn_v_b[i] = makeTensorHandle(model, layer.attn_v_b);
        model.attn_o_w[i] = makeTensorHandle(model, layer.attn_o_w);
        model.mlp_norm_w[i] = makeTensorHandle(model, layer.mlp_norm_w);
        model.mlp_gate_w[i] = makeTensorHandle(model, layer.mlp_gate_w);
        model.mlp_up_w[i] = makeTensorHandle(model, layer.mlp_up_w);
        model.mlp_down_w[i] = makeTensorHandle(model, layer.mlp_down_w);
    }

    model.weights.attn_norm_w = model.attn_norm_w.data();
    model.weights.attn_q_w = model.attn_q_w.data();
    model.weights.attn_q_b = model.attn_q_b.data();
    model.weights.attn_k_w = model.attn_k_w.data();
    model.weights.attn_k_b = model.attn_k_b.data();
    model.weights.attn_v_w = model.attn_v_w.data();
    model.weights.attn_v_b = model.attn_v_b.data();
    model.weights.attn_o_w = model.attn_o_w.data();
    model.weights.mlp_norm_w = model.mlp_norm_w.data();
    model.weights.mlp_gate_w = model.mlp_gate_w.data();
    model.weights.mlp_up_w = model.mlp_up_w.data();
    model.weights.mlp_down_w = model.mlp_down_w.data();
}

void reportQwen2Error(const char *api, const std::exception &error) noexcept {
    std::cerr << "[ERROR] " << api << " failed: " << error.what() << std::endl;
}

void reportUnknownQwen2Error(const char *api) noexcept {
    std::cerr << "[ERROR] " << api << " failed with an unknown exception." << std::endl;
}

} // namespace

__C {

    LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        if (meta == nullptr || device_ids == nullptr || ndevice != 1) {
            return nullptr;
        }

        try {
            std::vector<int> ids(device_ids, device_ids + ndevice);
            auto result = std::make_unique<LlaisysQwen2Model>();
            result->model = std::make_unique<llaisys::models::qwen2::Qwen2Model>(
                *meta,
                device,
                std::move(ids));
            buildWeightHandles(*result);
            return result.release();
        } catch (const std::exception &error) {
            reportQwen2Error("llaisysQwen2ModelCreate", error);
            return nullptr;
        } catch (...) {
            reportUnknownQwen2Error("llaisysQwen2ModelCreate");
            return nullptr;
        }
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
        delete model;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        return model == nullptr ? nullptr : &model->weights;
    }

    void llaisysQwen2ModelReset(LlaisysQwen2Model *model) {
        if (model != nullptr && model->model != nullptr) {
            model->model->resetCache();
        }
    }

    int64_t llaisysQwen2ModelInfer(
        LlaisysQwen2Model *model,
        int64_t *token_ids,
        size_t ntoken) {
        if (model == nullptr || model->model == nullptr || token_ids == nullptr || ntoken == 0) {
            return -1;
        }

        try {
            return model->model->infer(token_ids, ntoken);
        } catch (const std::exception &error) {
            reportQwen2Error("llaisysQwen2ModelInfer", error);
            return -1;
        } catch (...) {
            reportUnknownQwen2Error("llaisysQwen2ModelInfer");
            return -1;
        }
    }
}
