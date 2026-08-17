#include "model.hpp"

#include "../../utils.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include "../../core/llaisys_core.hpp"
#include <algorithm>

#include <cmath>
#include <utility>

namespace llaisys::models::qwen2 {

namespace {

tensor_t createWeight(
    const std::vector<size_t> &shape,
    const LlaisysQwen2Meta &meta,
    llaisysDeviceType_t device_type,
    int device_id) {
    return Tensor::create(shape, meta.dtype, device_type, device_id);
}

Qwen2LayerWeights createLayerWeights(
    const LlaisysQwen2Meta &meta,
    llaisysDeviceType_t device_type,
    int device_id) {
    const size_t q_size = meta.nh * meta.dh;
    const size_t kv_size = meta.nkvh * meta.dh;

    return Qwen2LayerWeights{
        createWeight({meta.hs}, meta, device_type, device_id),
        createWeight({q_size, meta.hs}, meta, device_type, device_id),
        createWeight({q_size}, meta, device_type, device_id),
        createWeight({kv_size, meta.hs}, meta, device_type, device_id),
        createWeight({kv_size}, meta, device_type, device_id),
        createWeight({kv_size, meta.hs}, meta, device_type, device_id),
        createWeight({kv_size}, meta, device_type, device_id),
        createWeight({meta.hs, q_size}, meta, device_type, device_id),
        createWeight({meta.hs}, meta, device_type, device_id),
        createWeight({meta.di, meta.hs}, meta, device_type, device_id),
        createWeight({meta.di, meta.hs}, meta, device_type, device_id),
        createWeight({meta.hs, meta.di}, meta, device_type, device_id),
    };
}

} // namespace

Qwen2Model::Qwen2Model(
    const LlaisysQwen2Meta &meta,
    llaisysDeviceType_t device_type,
    std::vector<int> device_ids)
    : _meta(meta),
      _device_type(device_type),
      _device_ids(std::move(device_ids)) {
    CHECK_ARGUMENT(_meta.nlayer > 0, "Qwen2: nlayer must be positive.");
    CHECK_ARGUMENT(_meta.hs > 0, "Qwen2: hidden size must be positive.");
    CHECK_ARGUMENT(_meta.nh > 0, "Qwen2: attention head count must be positive.");
    CHECK_ARGUMENT(_meta.nkvh > 0, "Qwen2: KV head count must be positive.");
    CHECK_ARGUMENT(_meta.dh > 0, "Qwen2: head dimension must be positive.");
    CHECK_ARGUMENT(_meta.di > 0, "Qwen2: intermediate size must be positive.");
    CHECK_ARGUMENT(_meta.maxseq > 0, "Qwen2: maximum sequence length must be positive.");
    CHECK_ARGUMENT(_meta.voc > 0, "Qwen2: vocabulary size must be positive.");
    CHECK_ARGUMENT(_meta.hs == _meta.nh * _meta.dh, "Qwen2: hidden size must equal nh * dh.");
    CHECK_ARGUMENT(_meta.nh % _meta.nkvh == 0, "Qwen2: attention heads must be divisible by KV heads.");
    CHECK_ARGUMENT(_meta.epsilon >= 0.0f, "Qwen2: RMSNorm epsilon must not be negative.");
    CHECK_ARGUMENT(_meta.theta > 0.0f, "Qwen2: RoPE theta must be positive.");
    CHECK_ARGUMENT(_device_ids.size() == 1, "Qwen2: exactly one device id is currently supported.");

    const int device_id = primaryDeviceId();

    // Context prefers an available accelerator when it is first initialized.
    // Explicitly activate the model's device before allocating weights; otherwise
    // a CPU model created from a CUDA-enabled build would allocate every weight as
    // CUDA pinned host memory through Tensor::create's host-tensor path.
    core::context().setDevice(_device_type, device_id);

    _weights.in_embed = createWeight({_meta.voc, _meta.hs}, _meta, _device_type, device_id);
    _weights.out_embed = createWeight({_meta.voc, _meta.hs}, _meta, _device_type, device_id);
    _weights.out_norm_w = createWeight({_meta.hs}, _meta, _device_type, device_id);

    _weights.layers.reserve(_meta.nlayer);
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _weights.layers.push_back(createLayerWeights(_meta, _device_type, device_id));
    }

    _kv_caches.resize(_meta.nlayer);
}

const LlaisysQwen2Meta &Qwen2Model::meta() const {
    return _meta;
}

llaisysDeviceType_t Qwen2Model::deviceType() const {
    return _device_type;
}

int Qwen2Model::primaryDeviceId() const {
    return _device_ids.front();
}

Qwen2Weights &Qwen2Model::weights() {
    return _weights;
}

std::vector<Qwen2KVCache> &Qwen2Model::kvCaches() {
    return _kv_caches;
}

void Qwen2Model::resetCache() {
    for (auto &cache : _kv_caches) {
        cache.length = 0;
    }
}

tensor_t Qwen2Model::embedTokens(
    const int64_t *token_ids,
    size_t ntoken) const {
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen2: token_ids must not be null.");
    CHECK_ARGUMENT(ntoken > 0, "Qwen2: ntoken must be positive.");

    auto indices = Tensor::create(
        {ntoken},
        LLAISYS_DTYPE_I64,
        _device_type,
        primaryDeviceId());
    indices->load(token_ids);

    auto hidden = Tensor::create(
        {ntoken, _meta.hs},
        _meta.dtype,
        _device_type,
        primaryDeviceId());

    ops::embedding(hidden, indices, _weights.in_embed);
    return hidden;
}

tensor_t Qwen2Model::createPositionIds(
    size_t start,
    size_t ntoken) const {
    CHECK_ARGUMENT(
        start + ntoken <= _meta.maxseq,
        "Qwen2: token positions exceed maximum sequence length.");

    std::vector<int64_t> positions(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        positions[i] = static_cast<int64_t>(start + i);
    }

    auto position_ids = Tensor::create(
        {ntoken},
        LLAISYS_DTYPE_I64,
        _device_type,
        primaryDeviceId());
    position_ids->load(positions.data());
    return position_ids;
}

void Qwen2Model::ensureCacheCapacity(
    Qwen2KVCache &cache,
    size_t required) {
    CHECK_ARGUMENT(
        required <= _meta.maxseq,
        "Qwen2: KV cache exceeds maximum sequence length.");

    if (cache.capacity >= required) {
        return;
    }

    size_t new_capacity = cache.capacity == 0
                              ? std::min(size_t(16), _meta.maxseq)
                              : cache.capacity;

    while (new_capacity < required) {
        new_capacity = std::min(
            new_capacity * 2,
            _meta.maxseq);
    }

    auto new_k = Tensor::create(
        {new_capacity, _meta.nkvh, _meta.dh},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    auto new_v = Tensor::create(
        {new_capacity, _meta.nkvh, _meta.dh},
        _meta.dtype,
        _device_type,
        primaryDeviceId());

    core::context().setDevice(
        _device_type,
        primaryDeviceId());

    if (cache.length > 0) {
        const size_t cache_bytes = cache.length
                                   * _meta.nkvh
                                   * _meta.dh
                                   * new_k->elementSize();

        core::context().runtime().api()->memcpy_sync(
            new_k->data(),
            cache.k->data(),
            cache_bytes,
            LLAISYS_MEMCPY_D2D);
        core::context().runtime().api()->memcpy_sync(
            new_v->data(),
            cache.v->data(),
            cache_bytes,
            LLAISYS_MEMCPY_D2D);
    }

    cache.k = std::move(new_k);
    cache.v = std::move(new_v);
    cache.capacity = new_capacity;
}

void Qwen2Model::appendKvCache(
    Qwen2KVCache &cache,
    tensor_t new_k,
    tensor_t new_v) {
    const size_t ntoken = new_k->shape()[0];
    const std::vector<size_t> expected_shape{
        ntoken,
        _meta.nkvh,
        _meta.dh,
    };

    CHECK_ARGUMENT(
        new_k->shape() == expected_shape,
        "Qwen2: new K cache shape is invalid.");
    CHECK_ARGUMENT(
        new_v->shape() == new_k->shape(),
        "Qwen2: new V cache shape is invalid.");

    const size_t required = cache.length + ntoken;
    ensureCacheCapacity(cache, required);

    const size_t bytes_per_token = _meta.nkvh
                                   * _meta.dh
                                   * new_k->elementSize();
    const size_t offset = cache.length * bytes_per_token;
    const size_t new_bytes = ntoken * bytes_per_token;

    core::context().setDevice(
        _device_type,
        primaryDeviceId());

    core::context().runtime().api()->memcpy_sync(
        cache.k->data() + offset,
        new_k->data(),
        new_bytes,
        LLAISYS_MEMCPY_D2D);
    core::context().runtime().api()->memcpy_sync(
        cache.v->data() + offset,
        new_v->data(),
        new_bytes,
        LLAISYS_MEMCPY_D2D);

    cache.length = required;
}

tensor_t Qwen2Model::forwardAttention(
    tensor_t hidden,
    tensor_t position_ids,
    const Qwen2LayerWeights &weights,
    Qwen2KVCache &cache) {
    CHECK_ARGUMENT(
        hidden->ndim() == 2 && hidden->shape()[1] == _meta.hs,
        "Qwen2: attention input shape is invalid.");
    CHECK_ARGUMENT(
        position_ids->ndim() == 1
            && position_ids->shape()[0] == hidden->shape()[0],
        "Qwen2: position id shape is invalid.");

    const size_t ntoken = hidden->shape()[0];
    const size_t q_size = _meta.nh * _meta.dh;
    const size_t kv_size = _meta.nkvh * _meta.dh;

    auto normalized = Tensor::create(
        {ntoken, _meta.hs},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    ops::rms_norm(
        normalized,
        hidden,
        weights.attn_norm_w,
        _meta.epsilon);

    auto q = Tensor::create(
        {ntoken, q_size},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    auto k = Tensor::create(
        {ntoken, kv_size},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    auto v = Tensor::create(
        {ntoken, kv_size},
        _meta.dtype,
        _device_type,
        primaryDeviceId());

    ops::linear(q, normalized, weights.attn_q_w, weights.attn_q_b);
    ops::linear(k, normalized, weights.attn_k_w, weights.attn_k_b);
    ops::linear(v, normalized, weights.attn_v_w, weights.attn_v_b);

    auto q_heads = q->view({ntoken, _meta.nh, _meta.dh});
    auto k_heads = k->view({ntoken, _meta.nkvh, _meta.dh});
    auto v_heads = v->view({ntoken, _meta.nkvh, _meta.dh});

    auto rotated_q = Tensor::create(
        {ntoken, _meta.nh, _meta.dh},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    auto rotated_k = Tensor::create(
        {ntoken, _meta.nkvh, _meta.dh},
        _meta.dtype,
        _device_type,
        primaryDeviceId());

    ops::rope(
        rotated_q,
        q_heads,
        position_ids,
        _meta.theta);
    ops::rope(
        rotated_k,
        k_heads,
        position_ids,
        _meta.theta);

    appendKvCache(cache, rotated_k, v_heads);

    auto cached_k = cache.k->slice(0, 0, cache.length);
    auto cached_v = cache.v->slice(0, 0, cache.length);

    auto attention_value = Tensor::create(
        {ntoken, _meta.nh, _meta.dh},
        _meta.dtype,
        _device_type,
        primaryDeviceId());

    const float scale = 1.0f
                        / std::sqrt(static_cast<float>(_meta.dh));

    ops::self_attention(
        attention_value,
        rotated_q,
        cached_k,
        cached_v,
        scale);

    auto projected = Tensor::create(
        {ntoken, _meta.hs},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    ops::linear(
        projected,
        attention_value->view({ntoken, q_size}),
        weights.attn_o_w,
        nullptr);

    auto result = Tensor::create(
        {ntoken, _meta.hs},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    ops::add(result, hidden, projected);

    return result;
}

tensor_t Qwen2Model::forwardMlp(
    tensor_t hidden,
    const Qwen2LayerWeights &weights) const {
    CHECK_ARGUMENT(
        hidden->ndim() == 2 && hidden->shape()[1] == _meta.hs,
        "Qwen2: MLP input shape is invalid.");

    const size_t ntoken = hidden->shape()[0];

    auto normalized = Tensor::create(
        {ntoken, _meta.hs},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    ops::rms_norm(
        normalized,
        hidden,
        weights.mlp_norm_w,
        _meta.epsilon);

    auto gate = Tensor::create(
        {ntoken, _meta.di},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    auto up = Tensor::create(
        {ntoken, _meta.di},
        _meta.dtype,
        _device_type,
        primaryDeviceId());

    ops::linear(
        gate,
        normalized,
        weights.mlp_gate_w,
        nullptr);
    ops::linear(
        up,
        normalized,
        weights.mlp_up_w,
        nullptr);

    auto activated = Tensor::create(
        {ntoken, _meta.di},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    ops::swiglu(activated, gate, up);

    auto down = Tensor::create(
        {ntoken, _meta.hs},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    ops::linear(
        down,
        activated,
        weights.mlp_down_w,
        nullptr);

    auto result = Tensor::create(
        {ntoken, _meta.hs},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    ops::add(result, hidden, down);

    return result;
}

tensor_t Qwen2Model::forwardLayer(
    tensor_t hidden,
    tensor_t position_ids,
    const Qwen2LayerWeights &weights,
    Qwen2KVCache &cache) {
    auto attention_output = forwardAttention(
        hidden,
        position_ids,
        weights,
        cache);

    return forwardMlp(
        attention_output,
        weights);
}

int64_t Qwen2Model::selectNextToken(tensor_t hidden) const {
    CHECK_ARGUMENT(hidden != nullptr, "Qwen2: hidden must not be null.");
    CHECK_ARGUMENT(
        hidden->ndim() == 2 && hidden->shape()[1] == _meta.hs,
        "Qwen2: final hidden shape is invalid.");

    const size_t ntoken = hidden->shape()[0];
    auto last_hidden = hidden->slice(0, ntoken - 1, ntoken);

    auto normalized = Tensor::create(
        {1, _meta.hs},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    ops::rms_norm(
        normalized,
        last_hidden,
        _weights.out_norm_w,
        _meta.epsilon);

    auto logits = Tensor::create(
        {1, _meta.voc},
        _meta.dtype,
        _device_type,
        primaryDeviceId());
    ops::linear(
        logits,
        normalized,
        _weights.out_embed,
        nullptr);

    auto max_index = Tensor::create(
        {1},
        LLAISYS_DTYPE_I64,
        _device_type,
        primaryDeviceId());
    auto max_value = Tensor::create(
        {1},
        _meta.dtype,
        _device_type,
        primaryDeviceId());

    ops::argmax(
        max_index,
        max_value,
        logits->view({_meta.voc}));

    if (_device_type == LLAISYS_DEVICE_CPU) {
        return *reinterpret_cast<const int64_t *>(max_index->data());
    }

    int64_t token = 0;
    auto &runtime = core::context().runtime();
    runtime.synchronize();
    runtime.api()->memcpy_sync(
        &token,
        max_index->data(),
        sizeof(token),
        LLAISYS_MEMCPY_D2H);
    return token;
}

int64_t Qwen2Model::infer(
    const int64_t *token_ids,
    size_t ntoken) {
    const size_t past_length = _kv_caches.front().length;

    for (const auto &cache : _kv_caches) {
        CHECK_ARGUMENT(
            cache.length == past_length,
            "Qwen2: KV cache lengths are inconsistent.");
    }

    auto hidden = embedTokens(token_ids, ntoken);
    auto position_ids = createPositionIds(
        past_length,
        ntoken);

    for (size_t layer_index = 0;
         layer_index < _weights.layers.size();
         ++layer_index) {
        hidden = forwardLayer(
            hidden,
            position_ids,
            _weights.layers[layer_index],
            _kv_caches[layer_index]);
    }

    return selectNextToken(hidden);
}

} // namespace llaisys::models::qwen2
