from typing import Sequence

import contextlib
import ctypes
from ..libllaisys.llaisys_types import DataType
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
import json
from pathlib import Path
import ml_dtypes  # noqa: F401 -- import registers numpy's bfloat16 dtype, needed before safetensors reads BF16 weights
import safetensors
from ..libllaisys import LlaisysQwen2Meta, LlaisysQwen2Weights, LlaisysQwen2Model


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
            self.meta = LlaisysQwen2Meta(
                dtype=DataType.BF16,
                nlayer=config["num_hidden_layers"],
                hs=config["hidden_size"],
                nh=config["num_attention_heads"],
                nkvh=config["num_key_value_heads"],
                dh=config["hidden_size"] // config["num_attention_heads"],
                di=config["intermediate_size"],
                maxseq=2048,  # 上限，别用 config 里的 max_position_embeddings（太大，内存装不下）
                voc=config["vocab_size"],
                epsilon=config["rms_norm_eps"],
                theta=config["rope_theta"],
                end_token=config["eos_token_id"],
            )

        device_ids = (ctypes.c_int * 1)(0)  # 单卡
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), device.value, device_ids, 1
        )
        self.weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)

        w = self.weights.contents

        # 完整 safetensors 名 -> w 上的属性名
        GLOBAL_MAP = {
            "model.embed_tokens.weight": "in_embed",
            "lm_head.weight": "out_embed",
            "model.norm.weight": "out_norm_w",
        }

        # 去掉 "model.layers.{i}." 前缀后的 key -> w 上的属性名（数组，按 layer 索引）
        LAYER_MAP = {
            "input_layernorm.weight": "attn_norm_w",
            "self_attn.q_proj.weight": "attn_q_w",
            "self_attn.q_proj.bias": "attn_q_b",
            "self_attn.k_proj.weight": "attn_k_w",
            "self_attn.k_proj.bias": "attn_k_b",
            "self_attn.v_proj.weight": "attn_v_w",
            "self_attn.v_proj.bias": "attn_v_b",
            "self_attn.o_proj.weight": "attn_o_w",
            # Attention 后、MLP 前的 RMSNorm
            "post_attention_layernorm.weight": "mlp_norm_w",
            # MLP
            "mlp.gate_proj.weight": "mlp_gate_w",
            "mlp.up_proj.weight": "mlp_up_w",
            "mlp.down_proj.weight": "mlp_down_w",
        }

        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_.keys():
                arr = data_.get_tensor(name_)
                # TODO: 加载前没校验 arr.dtype/contiguity 是否匹配 C++ 侧分配的 meta.dtype，
                # 目前是直接信任这次真实权重跑通了的 raw memcpy。
                ptr = arr.ctypes.data_as(ctypes.c_void_p)

                if name_.startswith("model.layers."):
                    _ ,_, remain = name_.partition("model.layers.")
                    layer_idx,_,key = remain.partition(".")
                    layer_idx = int(layer_idx)
                    attr = LAYER_MAP[key]
                    handle = getattr(w, attr)[layer_idx]
                else:
                    attr = GLOBAL_MAP[name_]
                    handle = getattr(w, attr)

                LIB_LLAISYS.tensorLoad(handle, ptr)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        step_context=None,
    ):
        if max_new_tokens is None:
            max_new_tokens = 1024
        if len(inputs) == 0:
            raise ValueError("inputs cannot be empty")
        generated_tokens = list(inputs)

        # prefill/decode 共用一个循环：第一轮 current_input 是整个 prompt（prefill），
        # 之后每轮只喂上一步生成的那个 token（decode）。
        current_input = generated_tokens
        for step in range(max_new_tokens):
            token_array = (ctypes.c_int64 * len(current_input))(*current_input)
            # 让调用方（如 benchmark 脚本）包一层 context 分别计时 prefill/decode，
            # 而不需要 generate() 本身依赖任何 profiling 库。
            ctx = step_context(step, step == 0) if step_context else contextlib.nullcontext()
            with ctx:
                next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self.model, token_array, len(current_input)
                )
            generated_tokens.append(next_token)
            if next_token == self.meta.end_token:
                break
            current_input = [next_token]

        return generated_tokens
