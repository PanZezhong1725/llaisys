from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import DataType

from pathlib import Path
import ctypes
import json
import torch
import safetensors

class _Qwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", ctypes.c_int), ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t), ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t), ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t), ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t), ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float), ("end_token", ctypes.c_int64),
    ]


def _dtype(dtype):
    name = str(dtype).lower()
    if name in ("bfloat16", "bf16"):
        return DataType.BF16
    if name == "float16":
        return DataType.F16
    if name == "float32":
        return DataType.F32
    if name == "float64":
        return DataType.F64
    if name == "int64":
        return DataType.I64
    raise TypeError(f"Unsupported Qwen2 weight dtype: {dtype}")


LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [
    ctypes.POINTER(_Qwen2Meta), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int
]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = ctypes.c_void_p
LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [ctypes.c_void_p]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None
LIB_LLAISYS.llaisysQwen2ModelLoadWeight.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
    ctypes.c_int, ctypes.c_void_p
]
LIB_LLAISYS.llaisysQwen2ModelLoadWeight.restype = None
LIB_LLAISYS.llaisysQwen2ModelResetCache.argtypes = [ctypes.c_void_p]
LIB_LLAISYS.llaisysQwen2ModelResetCache.restype = None
LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = ctypes.c_int64

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        hidden = int(config["hidden_size"])
        heads = int(config["num_attention_heads"])
        kv_heads = int(config.get("num_key_value_heads", heads))
        self._meta = _Qwen2Meta(
            int(DataType.BF16 if config.get("torch_dtype") == "bfloat16" else DataType.F32),
            int(config["num_hidden_layers"]), hidden, heads, kv_heads,
            int(config.get("head_dim", hidden // heads)), int(config["intermediate_size"]),
            int(config["max_position_embeddings"]), int(config["vocab_size"]),
            float(config.get("rms_norm_eps", 1e-6)), float(config.get("rope_theta", 10000.0)),
            int(config.get("eos_token_id", 0)),
        )
        self.device = device
        device_id = ctypes.c_int(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self._meta), int(device), ctypes.byref(device_id), 1
        )

        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                tensor = data_.get_tensor(name_).contiguous()
                shape = (ctypes.c_size_t * tensor.ndim)(*tensor.shape)
                LIB_LLAISYS.llaisysQwen2ModelLoadWeight(
                    self._model, name_.encode("utf-8"), shape, tensor.ndim,
                    int(_dtype(str(tensor.dtype).replace("torch.", ""))),
                    ctypes.c_void_p(tensor.data_ptr())
                )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        LIB_LLAISYS.llaisysQwen2ModelResetCache(self._model)
        tokens = [int(x) for x in inputs]
        steps = 128 if max_new_tokens is None else int(max_new_tokens)
        for step in range(steps):
            current = tokens if step == 0 else [tokens[-1]]
            values = (ctypes.c_int64 * len(current))(*current)
            token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, values, len(current)))
            tokens.append(token)
            if token == self._meta.end_token:
                break
        return tokens

    def __del__(self):
        model = getattr(self, "_model", None)
        if model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
            self._model = None
