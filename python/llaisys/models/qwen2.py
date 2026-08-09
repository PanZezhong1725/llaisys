from typing import Sequence

import ctypes
import json
import mmap
import re
import struct
from pathlib import Path

from ..libllaisys import DataType, DeviceType, LIB_LLAISYS, LlaisysQwen2Meta


_SAFE_DTYPE_TO_LLAISYS = {
    "F32": DataType.F32,
    "F16": DataType.F16,
    "BF16": DataType.BF16,
}

_SAFE_DTYPE_SIZE = {"F32": 4, "F16": 2, "BF16": 2}


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self._model = None
        self._model_path = Path(model_path)
        if not self._model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self._model_path}")

        config_path = self._model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        tensor_files = sorted(self._model_path.glob("*.safetensors"))
        if not tensor_files:
            raise FileNotFoundError(
                f"No safetensors weights found in: {self._model_path}"
            )

        dtype = self._detect_weight_dtype(tensor_files)
        hidden_size = int(config["hidden_size"])
        num_heads = int(config["num_attention_heads"])
        head_dim = int(config.get("head_dim", hidden_size // num_heads))
        eos_token_ids = self._read_eos_token_ids(config)
        self._eos_token_ids = frozenset(eos_token_ids)
        self._max_sequence_length = int(config["max_position_embeddings"])

        if config.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(
                f"Unsupported Qwen2 activation: {config.get('hidden_act')}"
            )
        rope_parameters = config.get("rope_parameters") or {}
        rope_scaling = config.get("rope_scaling")
        if rope_scaling not in (None, {}):
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type not in (None, "default"):
                raise NotImplementedError(
                    f"Unsupported Qwen2 RoPE scaling: {rope_type}"
                )
        layer_types = config.get("layer_types", [])
        if config.get("use_sliding_window", False) or any(
            layer_type != "full_attention" for layer_type in layer_types
        ):
            raise NotImplementedError("Sliding-window attention is not supported yet")

        self._meta = LlaisysQwen2Meta(
            dtype=int(dtype),
            nlayer=int(config["num_hidden_layers"]),
            hs=hidden_size,
            nh=num_heads,
            nkvh=int(config["num_key_value_heads"]),
            dh=head_dim,
            di=int(config["intermediate_size"]),
            maxseq=self._max_sequence_length,
            voc=int(config["vocab_size"]),
            epsilon=float(config["rms_norm_eps"]),
            theta=float(
                config.get("rope_theta", rope_parameters.get("rope_theta", 10000.0))
            ),
            end_token=eos_token_ids[0],
        )

        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self._meta), int(device), device_ids, 1
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        try:
            self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(
                self._model
            ).contents
            self._load_weights(tensor_files)
        except Exception:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
            raise

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    @staticmethod
    def _read_header(path: Path):
        with path.open("rb") as file:
            header_length_data = file.read(8)
            if len(header_length_data) != 8:
                raise ValueError(f"Invalid safetensors header: {path}")
            header_length = struct.unpack("<Q", header_length_data)[0]
            header = json.loads(file.read(header_length))
        return header_length, header

    @classmethod
    def _detect_weight_dtype(cls, tensor_files):
        detected = set()
        for path in tensor_files:
            _, header = cls._read_header(path)
            for name, info in header.items():
                if name != "__metadata__" and info["dtype"] in _SAFE_DTYPE_TO_LLAISYS:
                    detected.add(info["dtype"])
        if len(detected) != 1:
            raise ValueError(
                f"Expected one floating-point weight dtype, found: {sorted(detected)}"
            )
        return _SAFE_DTYPE_TO_LLAISYS[detected.pop()]

    def _read_eos_token_ids(self, config):
        generation_path = self._model_path / "generation_config.json"
        generation_config = {}
        if generation_path.is_file():
            with generation_path.open("r", encoding="utf-8") as file:
                generation_config = json.load(file)
        value = generation_config.get("eos_token_id", config.get("eos_token_id"))
        if value is None:
            raise ValueError("Model config does not define eos_token_id")
        values = value if isinstance(value, list) else [value]
        if not values:
            raise ValueError("eos_token_id cannot be empty")
        return [int(token) for token in values]

    def _weight_handle(self, name):
        if name == "model.embed_tokens.weight":
            return self._weights.in_embed
        if name == "lm_head.weight":
            return self._weights.out_embed
        if name == "model.norm.weight":
            return self._weights.out_norm_w

        match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
        if match is None:
            return None
        layer = int(match.group(1))
        if layer >= self._meta.nlayer:
            raise ValueError(f"Weight layer index out of range: {name}")
        field_by_suffix = {
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
        field = field_by_suffix.get(match.group(2))
        return None if field is None else getattr(self._weights, field)[layer]

    @staticmethod
    def _tensor_shape(handle):
        ndim = int(LIB_LLAISYS.tensorGetNdim(handle))
        shape = (ctypes.c_size_t * ndim)()
        LIB_LLAISYS.tensorGetShape(handle, shape)
        return tuple(shape)

    def _load_raw_tensor(self, handle, info, mapped_file, data_start, name):
        expected_shape = self._tensor_shape(handle)
        actual_shape = tuple(int(value) for value in info["shape"])
        if actual_shape != expected_shape:
            raise ValueError(
                f"Weight shape mismatch for {name}: "
                f"expected {expected_shape}, got {actual_shape}"
            )
        expected_dtype = int(LIB_LLAISYS.tensorGetDataType(handle))
        actual_dtype = _SAFE_DTYPE_TO_LLAISYS.get(info["dtype"])
        if actual_dtype is None or int(actual_dtype) != expected_dtype:
            raise ValueError(
                f"Weight dtype mismatch for {name}: {info['dtype']}"
            )

        begin, end = (int(value) for value in info["data_offsets"])
        size = end - begin
        expected_size = _SAFE_DTYPE_SIZE[info["dtype"]]
        for dimension in actual_shape:
            expected_size *= dimension
        if size != expected_size:
            raise ValueError(
                f"Weight byte size mismatch for {name}: "
                f"expected {expected_size}, got {size}"
            )
        buffer = ctypes.c_char.from_buffer(mapped_file, data_start + begin)
        try:
            LIB_LLAISYS.tensorLoad(handle, ctypes.addressof(buffer))
        finally:
            del buffer

    def _expected_weight_names(self):
        names = {"model.embed_tokens.weight", "model.norm.weight"}
        suffixes = (
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        )
        for layer in range(self._meta.nlayer):
            names.update(f"model.layers.{layer}.{suffix}" for suffix in suffixes)
        return names

    def _load_weights(self, tensor_files):
        has_lm_head = False
        for path in tensor_files:
            _, header = self._read_header(path)
            has_lm_head = has_lm_head or "lm_head.weight" in header

        loaded = set()
        for path in tensor_files:
            header_length, header = self._read_header(path)
            with path.open("rb") as file:
                with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY) as mapped:
                    data_start = 8 + header_length
                    for name, info in header.items():
                        if name == "__metadata__":
                            continue
                        handle = self._weight_handle(name)
                        if handle is None:
                            continue
                        self._load_raw_tensor(handle, info, mapped, data_start, name)
                        loaded.add(name)
                        if name == "model.embed_tokens.weight" and not has_lm_head:
                            self._load_raw_tensor(
                                self._weights.out_embed,
                                info,
                                mapped,
                                data_start,
                                "tied lm_head.weight",
                            )

        missing = self._expected_weight_names() - loaded
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            raise ValueError(f"Missing Qwen2 weights: {preview}")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        del top_p, temperature
        if top_k != 1:
            raise NotImplementedError("Qwen2 currently supports greedy decoding only")
        tokens = [int(token) for token in inputs]
        if not tokens:
            raise ValueError("At least one input token is required")
        if max_new_tokens is None:
            max_new_tokens = 128
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if len(tokens) + max_new_tokens > self._max_sequence_length:
            raise ValueError("Requested generation exceeds maximum context length")
        if max_new_tokens == 0:
            return tokens

        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)
        prompt = (ctypes.c_int64 * len(tokens))(*tokens)
        next_token = int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, prompt, len(tokens)
            )
        )
        if next_token < 0:
            raise RuntimeError("Qwen2 inference failed")
        tokens.append(next_token)

        for _ in range(max_new_tokens - 1):
            if next_token in self._eos_token_ids:
                break
            current = ctypes.c_int64(next_token)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model, ctypes.byref(current), 1
                )
            )
            if next_token < 0:
                raise RuntimeError("Qwen2 inference failed")
            tokens.append(next_token)
        return tokens
