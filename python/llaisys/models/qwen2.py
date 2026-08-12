from typing import Sequence
import ctypes
import json
import mmap
import struct
from ctypes import c_int, c_int64, c_size_t
from pathlib import Path
import numpy as np

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.qwen2 import LlaisysQwen2Meta


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        with (model_path / "config.json").open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        hidden_size = int(config["hidden_size"])
        head_count = int(config["num_attention_heads"])
        head_dim = int(config.get("head_dim", hidden_size // head_count))
        dtype_name = str(config.get("torch_dtype", "bfloat16")).lower()
        dtype = {"float32": DataType.F32, "float16": DataType.F16, "bfloat16": DataType.BF16}[dtype_name]
        eos = config.get("eos_token_id", 2)
        if isinstance(eos, list):
            eos = eos[0]

        meta = LlaisysQwen2Meta(
            int(dtype),
            int(config["num_hidden_layers"]),
            hidden_size,
            head_count,
            int(config.get("num_key_value_heads", head_count)),
            head_dim,
            int(config["intermediate_size"]),
            int(config.get("max_position_embeddings", 32768)),
            int(config["vocab_size"]),
            float(config.get("rms_norm_eps", 1e-6)),
            float(config.get("rope_theta", 10000.0)),
            int(eos),
        )
        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), int(device), device_ids, 1
        )
        if not self._model:
            raise RuntimeError("Unable to create Qwen2 model")
        self._end_token = int(eos)

        for file in sorted(model_path.glob("*.safetensors")):
            self._load_safetensors(file)

    def _load_safetensors(self, path):
        dtype_info = {
            "F32": (DataType.F32, np.dtype("<f4")),
            "F16": (DataType.F16, np.dtype("<f2")),
            # NumPy has no built-in bfloat16; uint16 preserves its raw bits.
            "BF16": (DataType.BF16, np.dtype("<u2")),
        }
        with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            header_size = struct.unpack_from("<Q", mapped, 0)[0]
            data_start = 8 + header_size
            header = json.loads(mapped[8:data_start].decode("utf-8"))
            for name, descriptor in header.items():
                if name == "__metadata__":
                    continue
                dtype_name = descriptor["dtype"]
                if dtype_name not in dtype_info:
                    raise TypeError(f"Unsupported Qwen2 weight dtype: {dtype_name}")
                tensor_dtype, numpy_dtype = dtype_info[dtype_name]
                shape_values = tuple(int(dim) for dim in descriptor["shape"])
                begin, end = descriptor["data_offsets"]
                count = int(np.prod(shape_values, dtype=np.int64))
                if end - begin != count * numpy_dtype.itemsize:
                    raise ValueError(f"Invalid safetensors byte range for {name}")
                values = np.frombuffer(mapped, dtype=numpy_dtype, count=count, offset=data_start + begin)
                shape = (c_size_t * len(shape_values))(*shape_values)
                LIB_LLAISYS.llaisysQwen2ModelLoadWeight(
                    self._model,
                    name.encode("utf-8"),
                    values.ctypes.data_as(ctypes.c_void_p),
                    shape,
                    c_size_t(len(shape_values)),
                    int(tensor_dtype),
                )
                del values

    def __del__(self):
        model = getattr(self, "_model", None)
        if model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        del top_k, top_p, temperature  # Assignment 3 uses deterministic argmax.
        if max_new_tokens is None:
            max_new_tokens = 128
        if max_new_tokens <= 0:
            return list(inputs)

        result = [int(token) for token in inputs]
        if not result:
            raise ValueError("inputs must contain at least one token")
        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)
        prompt = (c_int64 * len(result))(*result)
        next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, prompt, c_size_t(len(result))))
        for step in range(max_new_tokens):
            result.append(next_token)
            if next_token == self._end_token or step + 1 == max_new_tokens:
                break
            token = c_int64(next_token)
            next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, ctypes.byref(token), c_size_t(1)))
        return result
