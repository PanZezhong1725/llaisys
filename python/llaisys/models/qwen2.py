import ctypes
import json
import mmap
from ctypes import POINTER, byref, c_char, c_int, c_int64
from pathlib import Path
from typing import Sequence

from ..libllaisys import LIB_LLAISYS, DataType, DeviceType, LlaisysQwen2Meta


class _SafeTensorsFile:
    """Minimal safetensors reader.

    The safetensors numpy backend cannot represent bfloat16 and torch is not allowed
    here, so raw bytes are mapped out of the file and handed to tensorLoad, which
    copies them verbatim.
    """

    def __init__(self, path: Path):
        self._file = open(path, "rb")
        header_len = int.from_bytes(self._file.read(8), "little")
        self._header = json.loads(self._file.read(header_len))
        self._data_start = 8 + header_len
        # ACCESS_COPY keeps the mapping writable so from_buffer can alias it without
        # copying; nothing here ever writes to it.
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)

    def keys(self):
        return [k for k in self._header if k != "__metadata__"]

    def dtype(self, name: str) -> str:
        return self._header[name]["dtype"]

    def buffer(self, name: str):
        begin, end = self._header[name]["data_offsets"]
        return (c_char * (end - begin)).from_buffer(self._mm, self._data_start + begin)

    def close(self):
        self._mm.close()
        self._file.close()


class Qwen2:

    def __init__(
        self, model_path, device: DeviceType = DeviceType.CPU, max_seq: int = None
    ):
        model_path = Path(model_path)
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))

        nh = config["num_attention_heads"]
        hs = config["hidden_size"]

        # The config allows 131072 positions, which would make the KV cache far larger
        # than any test needs, so cap it.
        if max_seq is None:
            max_seq = min(config["max_position_embeddings"], 4096)

        end_token = config.get("eos_token_id", 151643)
        if isinstance(end_token, list):
            end_token = end_token[0]
        self._end_token = end_token

        self._meta = LlaisysQwen2Meta(
            dtype=DataType.BF16,
            nlayer=config["num_hidden_layers"],
            hs=hs,
            nh=nh,
            nkvh=config["num_key_value_heads"],
            dh=config.get("head_dim", hs // nh),
            di=config["intermediate_size"],
            maxseq=max_seq,
            voc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"],
            end_token=end_token,
        )

        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self._meta), c_int(device), device_ids, c_int(1)
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        targets = self._weight_targets(
            LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents
        )

        for file in sorted(model_path.glob("*.safetensors")):
            data_ = _SafeTensorsFile(file)
            try:
                for name_ in data_.keys():
                    target = targets.get(name_)
                    if target is None:
                        continue
                    if data_.dtype(name_) != "BF16":
                        raise RuntimeError(
                            f"{name_}: expected BF16, got {data_.dtype(name_)}"
                        )
                    buf = data_.buffer(name_)
                    LIB_LLAISYS.tensorLoad(target, ctypes.addressof(buf))
                    del buf  # release the mmap export before the file is closed
            finally:
                data_.close()

    def _weight_targets(self, weights):
        """Maps each safetensors name to the tensor handle it should be loaded into."""
        targets = {
            "model.embed_tokens.weight": weights.in_embed,
            "lm_head.weight": weights.out_embed,
            "model.norm.weight": weights.out_norm_w,
        }
        per_layer = {
            "input_layernorm.weight": weights.attn_norm_w,
            "self_attn.q_proj.weight": weights.attn_q_w,
            "self_attn.q_proj.bias": weights.attn_q_b,
            "self_attn.k_proj.weight": weights.attn_k_w,
            "self_attn.k_proj.bias": weights.attn_k_b,
            "self_attn.v_proj.weight": weights.attn_v_w,
            "self_attn.v_proj.bias": weights.attn_v_b,
            "self_attn.o_proj.weight": weights.attn_o_w,
            "post_attention_layernorm.weight": weights.mlp_norm_w,
            "mlp.gate_proj.weight": weights.mlp_gate_w,
            "mlp.up_proj.weight": weights.mlp_up_w,
            "mlp.down_proj.weight": weights.mlp_down_w,
        }
        for layer in range(self._meta.nlayer):
            for suffix, array in per_layer.items():
                targets[f"model.layers.{layer}.{suffix}"] = array[layer]
        return targets

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
        # Only argmax sampling is implemented, so top_k/top_p/temperature are unused.
        tokens = list(inputs)
        if max_new_tokens is None:
            max_new_tokens = self._meta.maxseq - len(tokens)

        LIB_LLAISYS.llaisysQwen2ModelResetCache(self._model)

        # Prefill the whole prompt, then feed one token per step; the backend KV cache
        # keeps earlier positions so each step only computes the new token.
        pending = tokens
        for _ in range(max_new_tokens):
            if len(tokens) >= self._meta.maxseq:
                break
            buf = (c_int64 * len(pending))(*pending)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model, ctypes.cast(buf, POINTER(c_int64)), len(pending)
                )
            )
            tokens.append(next_token)
            if next_token == self._end_token:
                break
            pending = [next_token]

        return tokens
