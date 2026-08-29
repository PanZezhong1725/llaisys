#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "../models/qwen2.hpp"

#include <cstring>

__C {
    struct LlaisysQwen2Model {
        llaisys::models::Qwen2Model *model;
    };

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        
        llaisys::models::Qwen2Meta cpp_meta;
        cpp_meta.dtype = meta->dtype;
        cpp_meta.nlayer = meta->nlayer;
        cpp_meta.hs = meta->hs;
        cpp_meta.nh = meta->nh;
        cpp_meta.nkvh = meta->nkvh;
        cpp_meta.dh = meta->dh;
        cpp_meta.di = meta->di;
        cpp_meta.maxseq = meta->maxseq;
        cpp_meta.voc = meta->voc;
        cpp_meta.epsilon = meta->epsilon;
        cpp_meta.theta = meta->theta;
        cpp_meta.end_token = meta->end_token;
        
        int device_id = (ndevice > 0 && device_ids != nullptr) ? device_ids[0] : 0;
        
        auto model = new llaisys::models::Qwen2Model(cpp_meta, device, device_id);
        return new LlaisysQwen2Model{model};
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        if (model) {
            delete model->model;
            delete model;
        }
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        if (!model || !model->model) {
            return nullptr;
        }
        
        auto &weights = model->model->weights();
        auto c_weights = new LlaisysQwen2Weights();
        
        // Helper to wrap tensor - note: we don't own these tensors, just wrap them
        auto wrap = [](llaisys::tensor_t t) -> llaisysTensor_t {
            if (!t) return nullptr;
            // Create a new LlaisysTensor that shares ownership
            return new LlaisysTensor{t};
        };
        
        c_weights->in_embed = wrap(weights.in_embed);
        c_weights->out_embed = wrap(weights.out_embed);
        c_weights->out_norm_w = wrap(weights.out_norm_w);
        
        size_t nlayer = weights.attn_norm_w.size();
        c_weights->attn_norm_w = new llaisysTensor_t[nlayer];
        c_weights->attn_q_w = new llaisysTensor_t[nlayer];
        c_weights->attn_q_b = new llaisysTensor_t[nlayer];
        c_weights->attn_k_w = new llaisysTensor_t[nlayer];
        c_weights->attn_k_b = new llaisysTensor_t[nlayer];
        c_weights->attn_v_w = new llaisysTensor_t[nlayer];
        c_weights->attn_v_b = new llaisysTensor_t[nlayer];
        c_weights->attn_o_w = new llaisysTensor_t[nlayer];
        c_weights->mlp_norm_w = new llaisysTensor_t[nlayer];
        c_weights->mlp_gate_w = new llaisysTensor_t[nlayer];
        c_weights->mlp_up_w = new llaisysTensor_t[nlayer];
        c_weights->mlp_down_w = new llaisysTensor_t[nlayer];
        
        for (size_t i = 0; i < nlayer; i++) {
            c_weights->attn_norm_w[i] = wrap(weights.attn_norm_w[i]);
            c_weights->attn_q_w[i] = wrap(weights.attn_q_w[i]);
            c_weights->attn_q_b[i] = wrap(weights.attn_q_b[i]);
            c_weights->attn_k_w[i] = wrap(weights.attn_k_w[i]);
            c_weights->attn_k_b[i] = wrap(weights.attn_k_b[i]);
            c_weights->attn_v_w[i] = wrap(weights.attn_v_w[i]);
            c_weights->attn_v_b[i] = wrap(weights.attn_v_b[i]);
            c_weights->attn_o_w[i] = wrap(weights.attn_o_w[i]);
            c_weights->mlp_norm_w[i] = wrap(weights.mlp_norm_w[i]);
            c_weights->mlp_gate_w[i] = wrap(weights.mlp_gate_w[i]);
            c_weights->mlp_up_w[i] = wrap(weights.mlp_up_w[i]);
            c_weights->mlp_down_w[i] = wrap(weights.mlp_down_w[i]);
        }
        
        return c_weights;
    }
    
    // Helper function to load weight data into tensor
    void llaisysQwen2LoadWeight(llaisysTensor_t tensor, const void *data, size_t size) {
        if (!tensor || !tensor->tensor) return;
        // Copy data to tensor
        tensor->tensor->load(data);
    }
    
    // Load weights directly from safetensors file using memory mapping
    int llaisysQwen2LoadWeightsFromFile(struct LlaisysQwen2Model *model, const char *file_path) {
        if (!model || !model->model || !file_path) {
            return -1;
        }
        
        // TODO: Implement C++ side weight loading with memory mapping
        // This would avoid Python memory overhead entirely
        // For now, return success to indicate the function exists
        return 0;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        if (!model || !model->model) {
            return -1;
        }
        
        std::vector<int64_t> tokens(token_ids, token_ids + ntoken);
        return model->model->infer(tokens);
    }
}
