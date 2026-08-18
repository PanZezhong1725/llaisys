from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import DataType
from ..libllaisys import LlaisysQwen2Meta

from pathlib import Path
import ctypes
import json
import re
import safetensors

# Cap on how many tokens (prompt + generated) a single model instance can
# hold in its KV cache. Real Qwen2 configs advertise max_position_embeddings
# in the tens of thousands, which would allocate a KV cache far larger than
# needed for a short prompt + a few hundred generated tokens.
_MAX_SEQ_LEN = 4096

_HF_DTYPE_TO_LLAISYS = {
    "bfloat16": DataType.BF16,
    "float16": DataType.F16,
    "float32": DataType.F32,
}

# safetensors name suffix (after "model.layers.{i}.") -> LlaisysQwen2Weights field name
_LAYER_FIELD_MAP = {
    "input_layernorm.weight": "attn_norm_w",
    "self_attn.q_proj.weight": "attn_q_w",
    "self_attn.q_proj.bias": "attn_q_b",
    "self_attn.k_proj.weight": "attn_k_w",
    "self_attn.k_proj.bias": "attn_k_b",
    "self_attn.v_proj.weight": "attn_v_w",
    "self_attn.v_proj.bias": "attn_v_b",
    "self_attn.o_proj.weight": "attn_o_w",
    "post_attention_layernorm.weight": "mlp_norm_w",
    "mlp.gate_proj.weight": "mlp_gate_w",
    "mlp.up_proj.weight": "mlp_up_w",
    "mlp.down_proj.weight": "mlp_down_w",
}

_LAYER_NAME_RE = re.compile(r"model\.layers\.(\d+)\.(.+)")


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        with open(model_path / "config.json", "r") as f:
            config = json.load(f)

        hidden_size = config["hidden_size"]
        num_attention_heads = config["num_attention_heads"]
        head_dim = config.get("head_dim", hidden_size // num_attention_heads)
        dtype = _HF_DTYPE_TO_LLAISYS.get(config.get("torch_dtype", "float32"), DataType.F32)

        eos_token_id = config["eos_token_id"]
        self._end_token = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id

        self._meta = LlaisysQwen2Meta(
            dtype=dtype,
            nlayer=config["num_hidden_layers"],
            hs=hidden_size,
            nh=num_attention_heads,
            nkvh=config["num_key_value_heads"],
            dh=head_dim,
            di=config["intermediate_size"],
            maxseq=min(config.get("max_position_embeddings", _MAX_SEQ_LEN), _MAX_SEQ_LEN),
            voc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"],
            end_token=self._end_token,
        )

        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self._meta), ctypes.c_int(device), device_ids, ctypes.c_int(1)
        )
        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents

        def load(handle, tensor):
            tensor = tensor.contiguous()
            LIB_LLAISYS.tensorLoad(handle, ctypes.c_void_p(tensor.data_ptr()))

        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                tensor = data_.get_tensor(name_)

                if name_ == "model.embed_tokens.weight":
                    load(weights.in_embed, tensor)
                    continue
                if name_ == "lm_head.weight":
                    load(weights.out_embed, tensor)
                    continue
                if name_ == "model.norm.weight":
                    load(weights.out_norm_w, tensor)
                    continue

                m = _LAYER_NAME_RE.match(name_)
                if not m:
                    continue
                layer_idx, suffix = int(m.group(1)), m.group(2)
                field = _LAYER_FIELD_MAP.get(suffix)
                if field is None:
                    continue
                load(getattr(weights, field)[layer_idx], tensor)

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        prompt_tokens = list(inputs)
        result_tokens = list(prompt_tokens)

        token_array = (ctypes.c_int64 * len(prompt_tokens))(*prompt_tokens)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model, token_array, ctypes.c_size_t(len(prompt_tokens))
        )
        result_tokens.append(next_token)

        steps = 1
        while next_token != self._end_token and steps < max_new_tokens:
            token_array = (ctypes.c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, token_array, ctypes.c_size_t(1)
            )
            result_tokens.append(next_token)
            steps += 1

        return result_tokens
