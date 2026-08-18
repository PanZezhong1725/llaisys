#include "llaisys/models/qwen2.h"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../tensor/tensor.hpp"
#include "../llaisys_tensor.hpp" // LlaisysTensor{tensor_t} 包装
#include <cmath>
// 头文件里 LlaisysQwen2Model 只是前向声明，"真身"定义在这里（不透明指针模式）。
struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights{};
    std::vector<llaisysTensor_t> attn_norm_w;
    std::vector<llaisysTensor_t> attn_q_w, attn_q_b;
    std::vector<llaisysTensor_t> attn_k_w, attn_k_b;
    std::vector<llaisysTensor_t> attn_v_w, attn_v_b;
    std::vector<llaisysTensor_t> attn_o_w;
    std::vector<llaisysTensor_t> mlp_norm_w;
    std::vector<llaisysTensor_t> mlp_gate_w, mlp_up_w, mlp_down_w;

    std::vector<llaisys::tensor_t> k_cache, v_cache;
    size_t cur_len = 0; // 已经缓存了多少个 token

    llaisysDeviceType_t device;
    std::vector<int> device_ids;
};

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        LlaisysQwen2Model *model = new LlaisysQwen2Model();
        model->meta = *meta;
        model->device = device;
        model->device_ids.assign(device_ids, device_ids + ndevice);

        int dev_id = (ndevice > 0 && device_ids) ? device_ids[0] : 0;
        size_t nlayer = meta->nlayer;

        // ---- single-tensor weights ----
        model->weights.in_embed = new LlaisysTensor{
            llaisys::Tensor::create({meta->voc, meta->hs}, meta->dtype, device, dev_id)};
        model->weights.out_embed = new LlaisysTensor{
            llaisys::Tensor::create({meta->voc, meta->hs}, meta->dtype, device, dev_id)};
        model->weights.out_norm_w = new LlaisysTensor{
            llaisys::Tensor::create({meta->hs}, meta->dtype, device, dev_id)};

        // ---- attn_norm_w ----
        model->attn_norm_w.resize(nlayer);
        for (size_t i = 0; i < nlayer; i++) {
            model->attn_norm_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->hs}, meta->dtype, device, dev_id)};
        }
        model->weights.attn_norm_w = model->attn_norm_w.data();

        // ---- attention q/k/v/o ----
        model->attn_q_w.resize(nlayer);
        model->attn_q_b.resize(nlayer);
        model->attn_k_w.resize(nlayer);
        model->attn_k_b.resize(nlayer);
        model->attn_v_w.resize(nlayer);
        model->attn_v_b.resize(nlayer);
        model->attn_o_w.resize(nlayer);
        for (size_t i = 0; i < nlayer; i++) {
            model->attn_q_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->nh * meta->dh, meta->hs}, meta->dtype, device, dev_id)};
            model->attn_q_b[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->nh * meta->dh}, meta->dtype, device, dev_id)};

            model->attn_k_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->nkvh * meta->dh, meta->hs}, meta->dtype, device, dev_id)};
            model->attn_k_b[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->nkvh * meta->dh}, meta->dtype, device, dev_id)};

            model->attn_v_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->nkvh * meta->dh, meta->hs}, meta->dtype, device, dev_id)};
            model->attn_v_b[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->nkvh * meta->dh}, meta->dtype, device, dev_id)};

            model->attn_o_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->hs, meta->nh * meta->dh}, meta->dtype, device, dev_id)};
        }
        model->weights.attn_q_w = model->attn_q_w.data();
        model->weights.attn_q_b = model->attn_q_b.data();
        model->weights.attn_k_w = model->attn_k_w.data();
        model->weights.attn_k_b = model->attn_k_b.data();
        model->weights.attn_v_w = model->attn_v_w.data();
        model->weights.attn_v_b = model->attn_v_b.data();
        model->weights.attn_o_w = model->attn_o_w.data();

        // ---- mlp norm/gate/up/down ----
        model->mlp_norm_w.resize(nlayer);
        model->mlp_gate_w.resize(nlayer);
        model->mlp_up_w.resize(nlayer);
        model->mlp_down_w.resize(nlayer);
        for (size_t i = 0; i < nlayer; i++) {
            model->mlp_norm_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->hs}, meta->dtype, device, dev_id)};
            model->mlp_gate_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->di, meta->hs}, meta->dtype, device, dev_id)};
            model->mlp_up_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->di, meta->hs}, meta->dtype, device, dev_id)};
            model->mlp_down_w[i] = new LlaisysTensor{
                llaisys::Tensor::create({meta->hs, meta->di}, meta->dtype, device, dev_id)};
        }
        model->weights.mlp_norm_w = model->mlp_norm_w.data();
        model->weights.mlp_gate_w = model->mlp_gate_w.data();
        model->weights.mlp_up_w = model->mlp_up_w.data();
        model->weights.mlp_down_w = model->mlp_down_w.data();

        // ---- KV cache (internal only, not exposed to Python) ----
        model->k_cache.resize(nlayer);
        model->v_cache.resize(nlayer);
        for (size_t i = 0; i < nlayer; i++) {
            model->k_cache[i] = llaisys::Tensor::create({meta->maxseq, meta->nkvh, meta->dh}, meta->dtype, device, dev_id);
            model->v_cache[i] = llaisys::Tensor::create({meta->maxseq, meta->nkvh, meta->dh}, meta->dtype, device, dev_id);
        }

        return model;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        delete model->weights.in_embed;
        delete model->weights.out_embed;
        delete model->weights.out_norm_w;
        for (size_t i = 0; i < model->meta.nlayer; i++) {
            delete model->attn_norm_w[i];
            delete model->attn_q_w[i];
            delete model->attn_q_b[i];
            delete model->attn_k_w[i];
            delete model->attn_k_b[i];
            delete model->attn_v_w[i];
            delete model->attn_v_b[i];
            delete model->attn_o_w[i];
            delete model->mlp_norm_w[i];
            delete model->mlp_gate_w[i];
            delete model->mlp_up_w[i];
            delete model->mlp_down_w[i];
        }
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        return &model->weights;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
        using llaisys::Tensor;
        using llaisys::tensor_t;
        namespace ops = llaisys::ops;

        const auto &meta = model->meta;
        auto device = model->device;
        auto device_id = (model->device_ids.size() > 0) ? model->device_ids[0] : 0;
        size_t cur_len = model->cur_len; // 已缓存的 token 数
        size_t total_len = cur_len + ntoken; // 这次调用结束后 cache 里的 token 数
        tensor_t input_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device, device_id);
        input_ids->load(token_ids);
        std::vector<int64_t> pos_ids_vec(ntoken);
        for (size_t i = 0; i < ntoken; i++) {
            pos_ids_vec[i] = cur_len + i;
        }
        tensor_t pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, device, device_id);
        pos_ids->load(pos_ids_vec.data());

        tensor_t input_embeds = Tensor::create({ntoken, meta.hs}, meta.dtype, device, device_id);
        ops::embedding(input_embeds, input_ids, model->weights.in_embed->tensor);

        float scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));
        for(size_t layer_idx = 0; layer_idx < meta.nlayer; layer_idx++) {
            tensor_t normed = Tensor::create({ntoken, meta.hs}, meta.dtype, device, device_id);
            ops::rms_norm(normed, input_embeds, model->weights.attn_norm_w[layer_idx]->tensor, meta.epsilon);

            // Q,K,V projections
            tensor_t q = Tensor::create({ntoken, meta.nh*meta.dh}, meta.dtype, device, device_id);
            tensor_t k = Tensor::create({ntoken, meta.nkvh*meta.dh}, meta.dtype, device, device_id);
            tensor_t v = Tensor::create({ntoken, meta.nkvh*meta.dh}, meta.dtype, device, device_id);
            ops::linear(q, normed, model->weights.attn_q_w[layer_idx]->tensor, model->weights.attn_q_b[layer_idx]->tensor);
            ops::linear(k, normed, model->weights.attn_k_w[layer_idx]->tensor, model->weights.attn_k_b[layer_idx]->tensor);
            ops::linear(v, normed, model->weights.attn_v_w[layer_idx]->tensor, model->weights.attn_v_b[layer_idx]->tensor);

            tensor_t q_reshaped = q->view({ntoken, meta.nh, meta.dh});
            tensor_t k_reshaped = k->view({ntoken, meta.nkvh, meta.dh});
            tensor_t v_reshaped = v->view({ntoken, meta.nkvh, meta.dh});

            tensor_t q_rope = Tensor::create({ntoken, meta.nh, meta.dh}, meta.dtype, device, device_id);
            tensor_t k_rope = Tensor::create({ntoken, meta.nkvh, meta.dh}, meta.dtype, device, device_id);
            ops::rope(q_rope, q_reshaped, pos_ids, meta.theta);
            ops::rope(k_rope, k_reshaped, pos_ids, meta.theta);

            // 写入本次新算出的 [cur_len, total_len) 段，再读出 [0, total_len) 完整历史
            model->k_cache[layer_idx]->slice(0,cur_len,total_len)->load(k_rope->data());
            model->v_cache[layer_idx]->slice(0,cur_len,total_len)->load(v_reshaped->data());
            tensor_t k_all = model->k_cache[layer_idx]->slice(0, 0, total_len);
            tensor_t v_all = model->v_cache[layer_idx]->slice(0, 0, total_len);

            tensor_t attn_val = Tensor::create({ntoken, meta.nh, meta.dh}, meta.dtype, device, device_id);
            ops::self_attention(attn_val, q_rope, k_all, v_all, scale);

            tensor_t attn_val_flat = attn_val->view({ntoken, meta.nh * meta.dh});
            tensor_t attn_out = Tensor::create({ntoken,meta.hs},meta.dtype,device,device_id);
            ops::linear(attn_out,attn_val_flat,model->weights.attn_o_w[layer_idx]->tensor,nullptr);
            tensor_t residual = Tensor::create({ntoken,meta.hs},meta.dtype,device,device_id);
            ops::add(residual,input_embeds, attn_out);

            // mlp
            tensor_t post_attention_normed = Tensor::create({ntoken,meta.hs},meta.dtype,device,device_id);
            ops::rms_norm(post_attention_normed, residual, model->weights.mlp_norm_w[layer_idx]->tensor, meta.epsilon);
            tensor_t after_swiglu =  Tensor::create({ntoken,meta.di},meta.dtype,device,device_id);
            tensor_t gate = Tensor::create({ntoken,meta.di},meta.dtype,device,device_id);
            tensor_t up = Tensor::create({ntoken,meta.di},meta.dtype,device,device_id);
            ops::linear(gate, post_attention_normed, model->weights.mlp_gate_w[layer_idx]->tensor, nullptr);
            ops::linear(up, post_attention_normed, model->weights.mlp_up_w[layer_idx]->tensor, nullptr);
            ops::swiglu(after_swiglu,gate,up);
            tensor_t after_down = Tensor::create({ntoken,meta.hs},meta.dtype,device,device_id);
            ops::linear(after_down,after_swiglu,model->weights.mlp_down_w[layer_idx]->tensor, nullptr);
            ops::add(input_embeds,after_down,residual);
        }
        tensor_t after_final_normed = Tensor::create({ntoken, meta.hs}, meta.dtype, device, device_id);
        ops::rms_norm(after_final_normed, input_embeds, model->weights.out_norm_w->tensor, meta.epsilon);
        tensor_t outputs_prob = Tensor::create({ntoken,meta.voc}, meta.dtype, device, device_id);
        ops::linear(outputs_prob,after_final_normed, model->weights.out_embed->tensor, nullptr);

        tensor_t last_logits = outputs_prob->slice(0, ntoken - 1, ntoken);
        tensor_t last_logits_1d = last_logits->view({meta.voc});
        tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
        tensor_t max_val = Tensor::create({1}, meta.dtype, device, device_id);
        ops::argmax(max_idx,max_val,last_logits_1d);

        model->cur_len = total_len;
        if (device == LLAISYS_DEVICE_CPU) {
            return *reinterpret_cast<int64_t *>(max_idx->data());
        } else {
            int64_t host_max_idx = 0;
            llaisys::core::context().runtime().api()->memcpy_sync(
                &host_max_idx,
                max_idx->data(),
                sizeof(int64_t),
                LLAISYS_MEMCPY_D2H
            );
            return host_max_idx;
        }
    }
}

