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

size_t qwen2DtypeSize(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        return 2;
    case LLAISYS_DTYPE_F32:
        return 4;
    default:
        throw std::invalid_argument("Unsupported Qwen2 cache dtype");
    }
}

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


void copyTensorBytes(
    llaisysTensor_t dst,
    llaisysTensor_t src,
    size_t bytes,
    llaisysDeviceType_t device,
    int device_id
) {
    if (bytes == 0) {
        return;
    }

    if (dst == nullptr || src == nullptr) {
        throw std::invalid_argument( "copyTensorBytes received null tensor");
    }

    llaisys::core::context().setDevice(device, device_id);

    llaisysMemcpyKind_t kind;

    if (device == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2H;
    } else {
        kind = LLAISYS_MEMCPY_D2D;
    }

    llaisys::core::context().runtime().api()->memcpy_sync(
        tensorGetData(dst),
        tensorGetData(src),
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

    std::vector<llaisysTensor_t> k_cache;
    std::vector<llaisysTensor_t> v_cache;

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

    int64_t prefill(
        const int64_t *token_ids,
        size_t ntoken
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


int64_t LlaisysQwen2Model::prefill(
    const int64_t *token_ids,
    size_t ntoken
) {
    if (token_ids == nullptr) {
        throw std::invalid_argument("token_ids must not be null");
    }

    if (ntoken == 0) {
        throw std::invalid_argument("ntoken must be greater zero");
    }

    if (ntoken > meta.maxseq) {
        throw std::length_error("Input sequence exceeds maxseq");
    }

    /*
     * 当前先完成 CPU 版本。
     *
     * GPU 版本最后读取 argmax 索引时需要 D2H，
     * 不能直接解引用设备指针。
     */

    if (device != LLAISYS_DEVICE_CPU) {
        throw std::runtime_error("Temporary prefill implementation only supports CPU");
    }

    for (size_t i = 0; i < ntoken; ++i) {
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
    auto input_ids = createTemporary({ntoken}, LLAISYS_DTYPE_I64);
    input_ids->load(token_ids);

    // 2. position IDS  无 kv cache，每次都从 0 开始：[0, 1, 2, ..., ntoken - 1]
    std::vector<int64_t> position_data(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        position_data[i] = static_cast<int64_t>(i);
    }
    auto position_ids = createTemporary({ntoken}, LLAISYS_DTYPE_I64);
    position_ids->load(position_data.data());

    // 3. Embedding
    //      input_ids: [seqlen]
    //      weight:    [vocab, hidden]
    //      hidden:    [seqlen, hidden]
    auto hidden = createTemporary({ntoken, meta.hs},  meta.dtype);
    llaisys::ops::embedding(
        hidden,
        input_ids,
        unwrapWeight(weights.in_embed, "model.embed_tokens.weight")
    );
    std::fprintf(
        stderr,
        "[Qwen2] embedding complete: seqlen=%zu hidden=%zu\n",
        ntoken, meta.hs
    );

    // 4. Decoder Layers
    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));
    for (size_t layer= 0; layer < meta.nlayer; ++layer) {
        std::fprintf( stderr, "[Qwen2] layer %zu/%zu\n", layer + 1, meta.nlayer);

        // a. Attention RMSNorm  [seqlen, hidden]
        auto attention_norm = createTemporary({ntoken, meta.hs},  meta.dtype);
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
        auto q_linear = createTemporary({ntoken, q_size}, meta.dtype);
        auto k_linear = createTemporary({ntoken, kv_size}, meta.dtype);
        auto v_linear = createTemporary({ntoken, kv_size}, meta.dtype);
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
        auto q = q_linear->view({ntoken, meta.nh, meta.dh});
        auto k = k_linear->view({ntoken, meta.nkvh, meta.dh});
        auto v = v_linear->view({ntoken, meta.nkvh, meta.dh});

        // d. ROPE 只作用于 Q 和 K
        auto q_rope = createTemporary({ntoken, meta.nh, meta.dh}, meta.dtype);
        auto k_rope = createTemporary({ntoken, meta.nkvh, meta.dh}, meta.dtype);
        llaisys::ops::rope(q_rope, q, position_ids, meta.theta);
        llaisys::ops::rope(k_rope, k, position_ids, meta.theta);

        // e. Causal GQA Self-Attention
        //      无 Cache：
        //      
        //       q_len     = ntoken
        //       total_len = ntoken
        //      
        //       Q:   [seqlen, nh, dh]
        //       K/V: [seqlen, nkvh, dh]
        //       Out: [seqlen, nh, dh]
        auto attention_value = createTemporary({ntoken, meta.nh, meta.dh}, meta.dtype);
        llaisys::ops::self_attention(attention_value, q_rope, k_rope, v, attention_scale);

        // f. 合并 heads：[seqlen, nh, dh] -> [seqlen, hidden]
        auto attention_flat = attention_value->view({ntoken, meta.hs});

        // g. O Projection  o_proj 没有 bias
        auto attention_projected = createTemporary({ntoken, meta.hs}, meta.dtype);
        llaisys::ops::linear(
            attention_projected,
            attention_flat,
            unwrapWeight(weights.attn_o_w[layer], "o_proj.weight"),
            nullptr
        );

        // h. 第一个残差连接  after_attention = hidden + attention_projected
        auto after_attention = createTemporary({ntoken, meta.hs}, meta.dtype);
        llaisys::ops::add(after_attention, hidden, attention_projected);

        // i. MLP RMSNorm
        auto mlp_norm = createTemporary({ntoken, meta.hs}, meta.dtype);
        llaisys::ops::rms_norm(
            mlp_norm,
            after_attention,
            unwrapWeight(weights.mlp_norm_w[layer], "post_attention_layernorm.weight"),
            meta.epsilon
        );

        // j. Gate/Up Projections  [seqlen, hidden] -> [seqlen, intermediate]
        auto gate = createTemporary({ntoken, meta.di}, meta.dtype);
        auto up = createTemporary({ntoken, meta.di}, meta.dtype);
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
        auto activated = createTemporary({ntoken, meta.di}, meta.dtype);
        llaisys::ops::swiglu(activated, gate, up);

        // l. Down Projection   [seqlen, intermediate] -> [seqlen, hidden]
        auto down = createTemporary({ntoken, meta.hs}, meta.dtype);
        llaisys::ops::linear(
            down,
            activated,
            unwrapWeight(weights.mlp_down_w[layer],"down_proj.weight"),
            nullptr
        );

        // m. 第二个残差连接  layer_output = after_attention  + down
        auto layer_output = createTemporary({ntoken, meta.hs}, meta.dtype);
        llaisys::ops::add(layer_output, after_attention, down);

        // n. 当前层输出成为下一层输入
        hidden = layer_output;

        std::fflush(stderr);
    }


    // 5. Final RMSNorm
    auto final_hidden = createTemporary({ntoken, meta.hs}, meta.dtype);
    llaisys::ops::rms_norm(
        final_hidden,
        hidden,
        unwrapWeight(weights.out_norm_w, "model.norm.weight"),
        meta.epsilon
    );

    // 6. 只取最后一个 token 的 hidden state
    // [seqlen, hidden] -> [1, hidden]
    auto last_hidden = final_hidden->slice(0,ntoken - 1,ntoken);
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

    std::fprintf(
        stderr,
        "[Qwen2] prefill complete: seqlen=%zu next_token=%lld\n",
        ntoken,
        static_cast<long long>(next_token)
    );

    std::fflush(stderr);

    return next_token;
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

    std::vector<llaisysTensor_t> new_k_cache(meta.nlayer, nullptr);
    std::vector<llaisysTensor_t> new_v_cache(meta.nlayer, nullptr);

    try {
        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            new_k_cache[layer] = createTensor({
                new_capacity,
                meta.nkvh,
                meta.dh,
            });

            new_v_cache[layer] = createTensor({
                new_capacity,
                meta.nkvh,
                meta.dh,
            });
        }

        // 只复制有效部分：[0, cache_len)
        // 未使用的 [cache_len, old_capacity) 无须复制
        if (cache_len > 0) {
            const size_t elements_per_token = meta.nkvh * meta.dh;
            const size_t bytes_per_token = elements_per_token * qwen2DtypeSize(meta.dtype);
            const size_t valid_bytes = cache_len * bytes_per_token;

            if (k_cache.size() != meta.nlayer || v_cache.size() != meta.nlayer) {
                throw std::runtime_error("Invalid existing KV cache layer count");
            }

            for (size_t layer = 0; layer < meta.nlayer; ++layer) {
                copyTensorBytes(
                    new_k_cache[layer],
                    k_cache[layer],
                    valid_bytes,
                    device,
                    device_id
                );

                copyTensorBytes(
                    new_v_cache[layer],
                    v_cache[layer],
                    valid_bytes,
                    device,
                    device_id
                );
            }
        }
    } catch (...) {
        // 扩容失败时，释放新 Cache，保留旧 Cache
        destroyTensorArray(new_k_cache);
        destroyTensorArray(new_v_cache);

        throw;
    }

    // 新 Cache 已成功创建并复制，才能释放旧 Cache
    destroyTensorArray(k_cache);
    destroyTensorArray(v_cache);

    k_cache = std::move(new_k_cache);
    v_cache = std::move(new_v_cache);

    cache_capacity = new_capacity;

    std::printf(
        "Qwen2 KV cache resized: "
        "capacity=%zu, valid_length=%zu\n",
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

    destroyTensorArray(k_cache);
    destroyTensorArray(v_cache);

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
        std::fprintf(
            stderr,
            "[Qwen2] inference failed: %s\n",
            error.what()
        );

        std::fflush(stderr);

        return -1;
    } catch (...) {
        std::fprintf(
            stderr,
            "[Qwen2] inference failed: unknown error\n"
        );

        std::fflush(stderr);

        return -1;
    }   
}
