#include "llaisys/models/qwen2.h"

#include "../../models/qwen2/qwen2.hpp"
#include "../llaisys_tensor.hpp"

#include <vector>

// Owns the C++ model plus the LlaisysTensor wrappers handed out through
// LlaisysQwen2Weights. The wrappers borrow the model's tensors, so callers must not
// destroy them; they die with the model.
struct LlaisysQwen2Model {
    llaisys::models::qwen2::Qwen2 model;
    LlaisysQwen2Weights weights{};
    std::vector<LlaisysTensor *> owned;
    std::vector<std::vector<llaisysTensor_t>> arrays;

    LlaisysQwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id)
        : model(meta, device, device_id) {}

    ~LlaisysQwen2Model() {
        for (auto *w : owned) {
            delete w;
        }
    }

    llaisysTensor_t wrap(const llaisys::tensor_t &t) {
        auto *w = new LlaisysTensor{t};
        owned.push_back(w);
        return w;
    }

    llaisysTensor_t *wrapAll(const std::vector<llaisys::tensor_t> &v) {
        arrays.emplace_back();
        auto &arr = arrays.back();
        arr.reserve(v.size());
        for (const auto &t : v) {
            arr.push_back(wrap(t));
        }
        return arr.data();
    }
};

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        const int device_id = (ndevice > 0 && device_ids != nullptr) ? device_ids[0] : 0;
        auto *m = new LlaisysQwen2Model(*meta, device, device_id);

        // arrays must not reallocate while wrapAll hands out pointers into it.
        m->arrays.reserve(12);

        auto &w = m->model.weights();
        m->weights.in_embed = m->wrap(w.in_embed);
        m->weights.out_embed = m->wrap(w.out_embed);
        m->weights.out_norm_w = m->wrap(w.out_norm_w);
        m->weights.attn_norm_w = m->wrapAll(w.attn_norm_w);
        m->weights.attn_q_w = m->wrapAll(w.attn_q_w);
        m->weights.attn_q_b = m->wrapAll(w.attn_q_b);
        m->weights.attn_k_w = m->wrapAll(w.attn_k_w);
        m->weights.attn_k_b = m->wrapAll(w.attn_k_b);
        m->weights.attn_v_w = m->wrapAll(w.attn_v_w);
        m->weights.attn_v_b = m->wrapAll(w.attn_v_b);
        m->weights.attn_o_w = m->wrapAll(w.attn_o_w);
        m->weights.mlp_norm_w = m->wrapAll(w.mlp_norm_w);
        m->weights.mlp_gate_w = m->wrapAll(w.mlp_gate_w);
        m->weights.mlp_up_w = m->wrapAll(w.mlp_up_w);
        m->weights.mlp_down_w = m->wrapAll(w.mlp_down_w);
        return m;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        return &model->weights;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
        return model->model.infer(token_ids, ntoken);
    }

    void llaisysQwen2ModelResetCache(struct LlaisysQwen2Model * model) {
        model->model.resetCache();
    }
}
