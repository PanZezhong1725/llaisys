#include "llaisys/models/qwen2.h"
#include "../../core/llaisys_core.hpp"
#include "../../tensor/tensor.hpp"
#include "../llaisys_tensor.hpp"
#include "llaisys.h"
#include "llaisys/runtime.h"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>
#include <stdexcept>
#include <memory>


namespace { // 只在当前文件可见

void destroyTensor(llaisysTensor_t &tensor) {
    if (tensor != nullptr) {
        tensorDestroy(tensor);
        tensor = nullptr;
    }
}


void destroyTensorArray(
    std::vector<llaisysTensor_t> &array
) {
    for (auto &tensor : array) {
        destroyTensor(tensor);
    }

    array.clear();
}


void copyTensor(
    const llaisys::tensor_t &dst,
    const llaisys::tensor_t &src
) {
    if (dst == nullptr || src == nullptr) {
        throw std::invalid_argument(
            "copyTensor received null tensor"
        );
    }

    if (dst->shape() != src->shape()) {
        throw std::invalid_argument(
            "copyTensor shape mismatch"
        );
    }

    if (dst->dtype() != src->dtype()) {
        throw std::invalid_argument(
            "copyTensor dtype mismatch"
        );
    }

    if (dst->deviceType() != src->deviceType() || dst->deviceId() != src->deviceId()) {
        throw std::invalid_argument(
            "copyTensor device mismatch"
        );
    }

    if (!dst->isContiguous() || !src->isContiguous()) {
        throw std::invalid_argument(
            "copyTensor requires contiguous tensors"
        );
    }

    const size_t bytes = src->numel() * src->elementSize();

    llaisysMemcpyKind_t kind;

    if (src->deviceType() == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2H;
    } else {
        kind = LLAISYS_MEMCPY_D2D;
    }

    llaisys::core::context().setDevice(src->deviceType(), src->deviceId());

    llaisys::core::context()
        .runtime()
        .api()
        ->memcpy_sync(
            dst->data(),
            src->data(),
            bytes,
            kind
        );
}


llaisys::tensor_t unwrapWeight(
    llaisysTensor_t handle,
    const char *name
) {
    if (handle == nullptr) {
        throw std::runtime_error(
            std::string("Null Qwen2 weight: ") + name
        );
    }

    if (handle->tensor == nullptr) {
        throw std::runtime_error(
            std::string("Empty Qwen2 weight: ") + name
        );
    }

    return handle->tensor;
}

} // namespace


struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    LlaisysQwen2Weights weights{};

    llaisysDeviceType_t device = LLAISYS_DEVICE_CPU;
    int device_id = 0;

    // 每个 Transformer Layer 都有一份独立的 input_layernorm.weight
    // 对于 28 层 Qwen2：
    //      model.layers.0.input_layernorm.weight
    //      model.layers.1.input_layernorm.weight
    //      ...
    //      model.layers.27.input_layernorm.weight
    // attn_norm_w 不是单个 Tensor，而是一个由 28 个 Tensor 句柄组成的数组
    //      attn_norm_w
    //          ├── [0] → 第 0 层的 RMSNorm 权重 Tensor
    //          ├── [1] → 第 1 层的 RMSNorm 权重 Tensor
    //          ├── [2] → 第 2 层的 RMSNorm 权重 Tensor
    //          └── ...

    // 下面这些是权重数组的真正所有者， 负责内存和生命周期
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

    // 每个 Decoder Layer 一份 K/V Cache
    std::vector<llaisys::tensor_t> k_cache;
    std::vector<llaisys::tensor_t> v_cache;

    size_t cache_len;       // 已经写入 Cache 的有效 token 数
    size_t cache_capacity;  // 当前 Cache 最多能容纳的 token 数

    llaisysTensor_t createTensor(
        std::initializer_list<size_t> shape
    ) const;

    void allocateTensorArray(
        std::vector<llaisysTensor_t> &array,
        std::initializer_list<size_t> shape
    );

    llaisys::tensor_t createTemporary(
        std::initializer_list<size_t> shape,
        llaisysDataType_t dtype
    ) const;

    int64_t forwardChunk(
        const int64_t *token_ids,
        size_t q_len
    );

    int64_t prefill(
        const int64_t *token_ids,
        size_t ntoken
    );

    int64_t decode(
        const int64_t token_id
    );


    void initWeights(); // 只创建 tenor 并绑定到 weights，权重加载是上层 python 完成的

    void bindWeightPointers();  // 绑定权重指针

    void ensureCacheCapacity(size_t required);  // 检查容量，分配新 cache，迁移旧 cache 数据，更新 cache_capacity

    void reset();   

    void destroy();

    ~LlaisysQwen2Model() {
        destroy();
    }
};


llaisysTensor_t LlaisysQwen2Model::createTensor(
    std::initializer_list<size_t> shape
) const {
    std::vector<size_t> dimensions(shape);

    llaisysTensor_t tensor = tensorCreate(
        dimensions.data(),
        dimensions.size(),
        meta.dtype,
        device,
        device_id
    );

    if (tensor == nullptr) {
        throw std::runtime_error("Failed to create Qwen2 tensor");
    }

    return tensor;
}


void LlaisysQwen2Model::allocateTensorArray(
    std::vector<llaisysTensor_t> &array,
    std::initializer_list<size_t> shape
) {
    array.resize(meta.nlayer, nullptr);

    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        array[layer] = createTensor(shape);
    }
}


llaisys::tensor_t LlaisysQwen2Model::createTemporary(
    std::initializer_list<size_t> shape,
    llaisysDataType_t dtype
) const {
    return llaisys::Tensor::create(
        std::vector<size_t>(shape),
        dtype,
        device,
        device_id
    );
}


int64_t LlaisysQwen2Model::forwardChunk(
    const int64_t *token_ids,
    size_t q_len
) {
    if (token_ids == nullptr) {
        throw std::invalid_argument("token_ids must not be null");
    }

    if (q_len == 0) {
        throw std::invalid_argument("q_len must be greater zero");
    }

    const size_t past_len = cache_len;
    if (q_len > meta.maxseq - past_len) {
        throw std::length_error("Input sequence exceeds maxseq");
    }

    const size_t total_len = past_len + q_len;
    ensureCacheCapacity(total_len);

    /*
     * 当前先完成 CPU 版本。
     *
     * GPU 版本最后读取 argmax 索引时需要 D2H，
     * 不能直接解引用设备指针。
     */

    if (device != LLAISYS_DEVICE_CPU) {
        throw std::runtime_error("Temporary prefill implementation only supports CPU");
    }

    for (size_t i = 0; i < q_len; ++i) {
        if (token_ids[i] < 0 || static_cast<size_t>(token_ids[i]) >= meta.voc) {
            throw std::out_of_range("Input token ID is outside vocabulary");
        } 
    }

    const size_t q_size = meta.nh * meta.dh;
    const size_t kv_size = meta.nkvh * meta.dh;

    if (q_size != meta.hs) {
        throw std::runtime_error("nh * dh must equal hidden size");
    }

    // 1. Token IDS  shapes:[seqlen]  dtype:I64
    auto input_ids = createTemporary({q_len}, LLAISYS_DTYPE_I64);
    input_ids->load(token_ids);

    // 2. position IDS 有 kv cache，每次都从 past_len 开始：[past_len + 0, past_len + 1, ..., past_len + q_len - 1]
    //
    // prefill:
    //      past_len = 0
    //      q_len = prompt_len
    //      position_ids = [0, 1, 2, ..., prompt_len - 1]
    //
    // decode:
    //      past_len = prompt_len
    //      q_len = 1
    //      position_ids = [prompt_len]
    std::vector<int64_t> position_data(q_len);
    for (size_t i = 0; i < q_len; ++i) {
        position_data[i] = static_cast<int64_t>(past_len + i);
    }
    auto position_ids = createTemporary({q_len}, LLAISYS_DTYPE_I64);
    position_ids->load(position_data.data());

    // 3. Embedding
    //      input_ids: [seqlen]
    //      weight:    [vocab, hidden]
    //      hidden:    [seqlen, hidden]
    auto hidden = createTemporary({q_len, meta.hs},  meta.dtype);
    llaisys::ops::embedding(
        hidden,
        input_ids,
        unwrapWeight(weights.in_embed, "model.embed_tokens.weight")
    );
    std::fprintf(
        stderr,
        "[Qwen2] embedding complete: seqlen=%zu hidden=%zu\n",
        q_len, meta.hs
    );

    // 4. Decoder Layers
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));
    for (size_t layer= 0; layer < meta.nlayer; ++layer) {
        std::fprintf( stderr, "[Qwen2] layer %zu/%zu\n", layer + 1, meta.nlayer);

        // a. Attention RMSNorm  [seqlen, hidden]
        auto attention_norm = createTemporary({q_len, meta.hs},  meta.dtype);
        llaisys::ops::rms_norm(
            attention_norm,
            hidden,
            unwrapWeight(weights.attn_norm_w[layer], "input_layernorm.weight"),
            meta.epsilon
        );

        // b. Q/K/V Projection
        //       Q: [seqlen, nh * dh]
        //       K: [seqlen, nkvh * dh]
        //       V: [seqlen, nkvh * dh]
        auto q_linear = createTemporary({q_len, q_size}, meta.dtype);
        auto k_linear = createTemporary({q_len, kv_size}, meta.dtype);
        auto v_linear = createTemporary({q_len, kv_size}, meta.dtype);
        llaisys::ops::linear(
            q_linear,
            attention_norm,
            unwrapWeight(weights.attn_q_w[layer], "q_proj.weight"),
            unwrapWeight(weights.attn_q_b[layer], "q_proj.bias")
        );
        llaisys::ops::linear(
            k_linear,
            attention_norm,
            unwrapWeight(weights.attn_k_w[layer], "k_proj.weight"),
            unwrapWeight(weights.attn_k_b[layer], "k_proj.bias")
        );
        llaisys::ops::linear(
            v_linear,
            attention_norm,
            unwrapWeight(weights.attn_v_w[layer], "v_proj.weight"),
            unwrapWeight(weights.attn_v_b[layer], "v_proj.bias")
        );

        // c. 将 Projection 输出分头
        //       Q: [seqlen, nh, dh]
        //       K: [seqlen, nkvh, dh]
        //       V: [seqlen, nkvh, dh]
        // Projection 输出连续，因此可以直接 view
        auto q = q_linear->view({q_len, meta.nh, meta.dh});
        auto k = k_linear->view({q_len, meta.nkvh, meta.dh});
        auto v = v_linear->view({q_len, meta.nkvh, meta.dh});

        // d. ROPE 只作用于 Q 和 K
        auto q_rope = createTemporary({q_len, meta.nh, meta.dh}, meta.dtype);
        auto k_rope = createTemporary({q_len, meta.nkvh, meta.dh}, meta.dtype);
        llaisys::ops::rope(q_rope, q, position_ids, meta.theta);
        llaisys::ops::rope(k_rope, k, position_ids, meta.theta);

        // 写入范围：
        //      Prefill: [0, prompt_len)
        //      Decode:  [cache_len, cache_len + 1)
        auto k_write = k_cache[layer]->slice(0, past_len, total_len);
        auto v_write = v_cache[layer]->slice(0, past_len, total_len);
        copyTensor(k_write, k_rope);    // Cache 保存 RoPE 后的 K
        copyTensor(v_write, v);         // V 不应用 RoPE

        // e. Causal GQA Self-Attention
        //    有 Cache：
        //
        //     prefill
        //       Q:   [prompt_len, nh, dh]
        //       K/V: [prompt_len, nkvh, dh]
        //     decode
        //       Q:   [1, nh, dh]
        //       K/V: [past_len + 1, nkvh, dh]
        //
        //       Out: [seqlen, nh, dh]
        auto k_total = k_cache[layer]->slice(0, 0, total_len);
        auto v_total = v_cache[layer]->slice(0, 0, total_len);
        auto attention_value = createTemporary({q_len, meta.nh, meta.dh}, meta.dtype);
        llaisys::ops::self_attention(attention_value, q_rope, k_total, v_total, attention_scale);

        // f. 合并 heads：[seqlen, nh, dh] -> [seqlen, hidden]
        auto attention_flat = attention_value->view({q_len, meta.hs});

        // g. O Projection  o_proj 没有 bias
        auto attention_projected = createTemporary({q_len, meta.hs}, meta.dtype);
        llaisys::ops::linear(
            attention_projected,
            attention_flat,
            unwrapWeight(weights.attn_o_w[layer], "o_proj.weight"),
            nullptr
        );

        // h. 第一个残差连接  after_attention = hidden + attention_projected
        auto after_attention = createTemporary({q_len, meta.hs}, meta.dtype);
        llaisys::ops::add(after_attention, hidden, attention_projected);

        // i. MLP RMSNorm
        auto mlp_norm = createTemporary({q_len, meta.hs}, meta.dtype);
        llaisys::ops::rms_norm(
            mlp_norm,
            after_attention,
            unwrapWeight(weights.mlp_norm_w[layer], "post_attention_layernorm.weight"),
            meta.epsilon
        );

        // j. Gate/Up Projections  [seqlen, hidden] -> [seqlen, intermediate]
        auto gate = createTemporary({q_len, meta.di}, meta.dtype);
        auto up = createTemporary({q_len, meta.di}, meta.dtype);
        llaisys::ops::linear(
            gate,
            mlp_norm,
            unwrapWeight(weights.mlp_gate_w[layer],"gate_proj.weight"),
            nullptr
        );
        llaisys::ops::linear(
            up,
            mlp_norm,
            unwrapWeight(weights.mlp_up_w[layer],"up_proj.weight"),
            nullptr
        );

        // k. SwiGLU: SiLU(gate) * up
        auto activated = createTemporary({q_len, meta.di}, meta.dtype);
        llaisys::ops::swiglu(activated, gate, up);

        // l. Down Projection   [seqlen, intermediate] -> [seqlen, hidden]
        auto down = createTemporary({q_len, meta.hs}, meta.dtype);
        llaisys::ops::linear(
            down,
            activated,
            unwrapWeight(weights.mlp_down_w[layer],"down_proj.weight"),
            nullptr
        );

        // m. 第二个残差连接  layer_output = after_attention  + down
        auto layer_output = createTemporary({q_len, meta.hs}, meta.dtype);
        llaisys::ops::add(layer_output, after_attention, down);

        // n. 当前层输出成为下一层输入
        hidden = layer_output;

        std::fflush(stderr);
    }


    // 5. Final RMSNorm
    auto final_hidden = createTemporary({q_len, meta.hs}, meta.dtype);
    llaisys::ops::rms_norm(
        final_hidden,
        hidden,
        unwrapWeight(weights.out_norm_w, "model.norm.weight"),
        meta.epsilon
    );

    // 6. 只取最后一个 token 的 hidden state
    // [seqlen, hidden] -> [1, hidden]
    auto last_hidden = final_hidden->slice(0, q_len - 1, q_len);
    if (!last_hidden->isContiguous()) {
        throw std::runtime_error("Last hidden state is not contiguous");
    }

    // 7. LM head   [1, hidden] × [vocab, hidden]^T -> [1, vocab]
    auto logits_2d = createTemporary({1, meta.voc}, meta.dtype);
    llaisys::ops::linear(
        logits_2d,
        last_hidden,
        unwrapWeight(weights.out_embed, "lm_head.weight"),
        nullptr
    );
    // Argmax 当前要求 vals 是 1D Tensor
    auto logits = logits_2d->view({meta.voc});
    auto max_index = createTemporary({1}, LLAISYS_DTYPE_I64);
    auto max_value = createTemporary({1}, meta.dtype);
    llaisys::ops::argmax(max_index, max_value, logits);

    const auto *index_data =reinterpret_cast<const int64_t *>(max_index->data());
    const int64_t next_token = index_data[0];
    if (next_token < 0 || static_cast<size_t>(next_token) >= meta.voc) {
        throw std::runtime_error("Argmax returned an invalid token ID");
    }

    cache_len = total_len;

    std::fprintf(
        stderr,
        "[Qwen2] forward complete: q_len=%zu past_len=%zu total_len=%zu next_token=%lld\n",
        q_len,
        past_len,
        total_len,
        static_cast<long long>(next_token)
    );

    std::fflush(stderr);

    return next_token;
}


int64_t LlaisysQwen2Model::prefill(
    const int64_t *token_ids,
    size_t ntoken
) {
    if (cache_len != 0) {
        throw std::runtime_error(
            "Prefill requires an empty KV Cache"
        );
    }

    return forwardChunk(
        token_ids,
        ntoken
    );
}


int64_t LlaisysQwen2Model::decode(
    int64_t token_id
) {
    if (cache_len == 0) {
        throw std::runtime_error(
            "Decode requires a non-empty KV Cache"
        );
    }

    return forwardChunk(
        &token_id,
        1
    );
}


void LlaisysQwen2Model::initWeights() {
    const size_t q_size = meta.nh * meta.dh;
    const size_t kv_size = meta.nkvh * meta.dh;

    // model.embed_tokens.weight: [vocab_size, hidden_size]
    weights.in_embed = createTensor({
        meta.voc,
        meta.hs,
    });

    // lm_head.weight: [vocab_size, hidden_size]
    weights.out_embed = createTensor({
        meta.voc,
        meta.hs,
    });

    // model.norm.weight: [hidden_size]
    weights.out_norm_w = createTensor({
        meta.hs,
    });

    // Hugging Face Linear 权重布局是 [out_features, in_features]

    // input_layernorm.weight: [hidden_size]
    allocateTensorArray(
        attn_norm_w,
        {meta.hs}
    );

    // q_proj.weight: [num_q_heads * head_dim, hidden_size]
    allocateTensorArray(
        attn_q_w,
        {q_size, meta.hs}
    );

    // q_proj.bias: [num_q_heads * head_dim]
    allocateTensorArray(
        attn_q_b,
        {q_size}
    );

    // k_proj.weight: [num_kv_heads * head_dim, hidden_size]
    allocateTensorArray(
        attn_k_w,
        {kv_size, meta.hs}
    );

    // k_proj.bias: [num_kv_heads * head_dim]
    allocateTensorArray(
        attn_k_b,
        {kv_size}
    );

    // v_proj.weight: [num_kv_heads * head_dim, hidden_size]
    allocateTensorArray(
        attn_v_w,
        {kv_size, meta.hs}
    );

    // v_proj.bias: [num_kv_heads * head_dim]
    allocateTensorArray(
        attn_v_b,
        {kv_size}
    );

    // o_proj.weight: [hidden_size, num_q_heads * head_dim]
    allocateTensorArray(
        attn_o_w,
        {meta.hs, q_size}
    );

    // post_attention_layernorm.weight: [hidden_size]
    allocateTensorArray(
        mlp_norm_w,
        {meta.hs}
    );

    // gate_proj.weight: [intermediate_size, hidden_size]
    allocateTensorArray(
        mlp_gate_w,
        {meta.di, meta.hs}
    );

    // up_proj.weight: [intermediate_size, hidden_size]
    allocateTensorArray(
        mlp_up_w,
        {meta.di, meta.hs}
    );

    // down_proj.weight: [hidden_size, intermediate_size]
    allocateTensorArray(
        mlp_down_w,
        {meta.hs, meta.di}
    );
    
    // 绑定公开的权重指针
    bindWeightPointers();
}
 
void LlaisysQwen2Model::bindWeightPointers() {
    weights.attn_norm_w = attn_norm_w.data();

    weights.attn_q_w = attn_q_w.data();
    weights.attn_q_b = attn_q_b.data();

    weights.attn_k_w = attn_k_w.data();
    weights.attn_k_b = attn_k_b.data();

    weights.attn_v_w = attn_v_w.data();
    weights.attn_v_b = attn_v_b.data();

    weights.attn_o_w = attn_o_w.data();

    weights.mlp_norm_w = mlp_norm_w.data();
    weights.mlp_gate_w = mlp_gate_w.data();
    weights.mlp_up_w = mlp_up_w.data();
    weights.mlp_down_w = mlp_down_w.data();
}


void LlaisysQwen2Model::ensureCacheCapacity(
    size_t required
) {
    if (required <= cache_capacity) {
        return;
    }

    // meta.maxseq ："max_position_embeddings": 131072
    // 限制：Prompt token 数 + 已进入 Cache 的生成 token 数
    if (required > meta.maxseq) {
        throw std::length_error(
            "Qwen2 KV cache exceeds max sequence length"
        );
    }

    // 第一次还没有 Cache 就从 256 开始
    // 每次默认扩容 256，不够就倍增
    size_t new_capacity =
        cache_capacity == 0
            ? std::min(size_t{256}, meta.maxseq)
            : cache_capacity;

    while (new_capacity < required) {
        // 防止乘 2 后超过 maxseq
        if (new_capacity > meta.maxseq / 2) {
            new_capacity = meta.maxseq;
            break;
        }

        new_capacity *= 2;
    }

    if (new_capacity < required) {
        throw std::length_error(
            "Unable to allocate enough KV cache capacity"
        );
    }

    std::vector<llaisys::tensor_t> new_k_cache(meta.nlayer);
    std::vector<llaisys::tensor_t> new_v_cache(meta.nlayer);

    
    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        new_k_cache[layer] = createTemporary(
            {
                new_capacity,
                meta.nkvh,
                meta.dh,
            },
            meta.dtype
        );

        new_v_cache[layer] = createTemporary(
            {
                new_capacity,
                meta.nkvh,
                meta.dh,
            },
            meta.dtype
        );

        // 只复制有效部分：[0, cache_len)
        // 未使用的 [cache_len, old_capacity) 无须复制
        if (cache_len > 0) {
            auto old_k = k_cache[layer]->slice(0, 0, cache_len);
            auto old_v = v_cache[layer]->slice(0, 0, cache_len);
            auto new_k = new_k_cache[layer]->slice(0, 0, cache_len);
            auto new_v = new_v_cache[layer]->slice(0, 0, cache_len);

            copyTensor(new_k, old_k);
            copyTensor(new_v, old_v);
        }
    }

    k_cache = std::move(new_k_cache);
    v_cache = std::move(new_v_cache);

    cache_capacity = new_capacity;

    std::printf(
        "Qwen2 KV cache resized: capacity=%zu, valid_length=%zu\n",
        cache_capacity,
        cache_len
    );
}


void LlaisysQwen2Model::destroy() {
    destroyTensor(weights.in_embed);
    destroyTensor(weights.out_embed);
    destroyTensor(weights.out_norm_w);

    destroyTensorArray(attn_norm_w);

    destroyTensorArray(attn_q_w);
    destroyTensorArray(attn_q_b);

    destroyTensorArray(attn_k_w);
    destroyTensorArray(attn_k_b);

    destroyTensorArray(attn_v_w);
    destroyTensorArray(attn_v_b);

    destroyTensorArray(attn_o_w);

    destroyTensorArray(mlp_norm_w);
    destroyTensorArray(mlp_gate_w);
    destroyTensorArray(mlp_up_w);
    destroyTensorArray(mlp_down_w);

    weights = {};

    cache_len = 0;
    cache_capacity = 0;
}


void LlaisysQwen2Model::reset() {
    cache_len = 0;
}


struct LlaisysQwen2Model *
llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice
) {
    if (meta == nullptr || device_ids == nullptr || ndevice != 1
    ) {
        std::fprintf(stderr,"Invalid arguments for Qwen2 model creation\n");
        return nullptr;
    }

    try {
        auto model = std::make_unique<LlaisysQwen2Model>();

        model->meta = *meta;
        model->device = device;
        model->device_id = device_ids[0];

        model->initWeights();

        return model.release();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Failed to create Qwen2 model: %s\n", error.what());
        return nullptr;
    }
}


void llaisysQwen2ModelDestroy(
    struct LlaisysQwen2Model *model
) {
    delete model;
}


struct LlaisysQwen2Weights * llaisysQwen2ModelWeights(
    struct LlaisysQwen2Model *model
) {
    if (model == nullptr) {
        return nullptr;
    }

    return &model->weights;
}


void llaisysQwen2ModelReset(
    struct LlaisysQwen2Model *model
) {
    // cache 原有数据不需要清零，后续 attention 只读取 [0, cache_len + q_len)
    // reset 后新 prompt 会从位置 0 覆盖原数据
    if (model != nullptr) {
        model->reset();
    }
}


int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken
) {
    if (model == nullptr || token_ids == nullptr || ntoken == 0) {
        std::fprintf(stderr, "llaisysQwen2ModelInfer: invalid arguments\n");
        return -1;
    }

    try {
        return model->prefill(token_ids, ntoken);
    } catch (const std::exception &error) {
       if (model->cache_len == 0) {
            return model->prefill(
                token_ids,
                ntoken
            );
        }

        if (ntoken != 1) {
            throw std::invalid_argument(
                "Decode requires exactly one token"
            );
        }

        return model->decode(token_ids[0]);
    } catch (...) {
        std::fprintf(
            stderr,
            "[Qwen2] inference failed: unknown error\n"
        );

        std::fflush(stderr);

        return -1;
    }   
}
