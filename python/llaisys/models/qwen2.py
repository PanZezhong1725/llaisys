from typing import Sequence, Optional
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import MemcpyKind
from ..libllaisys.llaisys_types import DataType
from ..libllaisys.tensor import llaisysTensor_t
from ..libllaisys.qwen2 import llaisysQwen2Model_t

from pathlib import Path
import safetensors.torch
import torch
import numpy as np
import ctypes
import math


QWEN2_CONFIG = {
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "num_hidden_layers": 28,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 131072,
    "vocab_size": 151936,
    "tie_word_embeddings": False,
}


def _create_tensor(shape, dtype, device_type=DeviceType.CPU, device_id=0):
    ndim = len(shape)
    shape_arr = (ctypes.c_size_t * ndim)(*shape)
    tensor = LIB_LLAISYS.tensorCreate(
        shape_arr,
        ctypes.c_size_t(ndim),
        ctypes.c_int(dtype.value),
        ctypes.c_int(device_type.value),
        ctypes.c_int(device_id),
    )
    return tensor


def _destroy_tensor(tensor):
    LIB_LLAISYS.tensorDestroy(tensor)


def _tensor_load(tensor, data):
    data_ptr = data.ctypes.data_as(ctypes.c_void_p)
    LIB_LLAISYS.tensorLoad(tensor, data_ptr)


def _tensor_get_data(tensor):
    return LIB_LLAISYS.tensorGetData(tensor)


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.device = device
        self.device_id = 0
        self.cfg = dict(QWEN2_CONFIG)

        self._model = LIB_LLAISYS.qwen2Create(
            ctypes.c_int(device.value),
            ctypes.c_int(self.device_id),
        )

        model_path = Path(model_path)
        self._load_weights(model_path)

    def _load_weights(self, model_path: Path):
        safetensor_files = sorted(model_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")

        for file in safetensor_files:
            print(f"Loading {file.name} ...")
            state_dict = safetensors.torch.load_file(str(file), device="cpu")
            for name_, tensor_torch in state_dict.items():
                if tensor_torch.dtype == torch.bfloat16:
                    tensor_np = tensor_torch.float().numpy()
                elif tensor_torch.dtype == torch.float16:
                    tensor_np = tensor_torch.float().numpy()
                elif tensor_torch.dtype == torch.float64:
                    tensor_np = tensor_torch.float().numpy()
                else:
                    tensor_np = tensor_torch.numpy()

                shape = list(tensor_np.shape)
                tensor = _create_tensor(shape, DataType.F32, self.device, self.device_id)
                _tensor_load(tensor, tensor_np)

                LIB_LLAISYS.qwen2LoadWeight(
                    self._model,
                    name_.encode("utf-8"),
                    tensor,
                )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> Sequence[int]:
        vocab_size = self.cfg["vocab_size"]

        if max_new_tokens is None:
            max_new_tokens = self.cfg["max_position_embeddings"] - len(inputs)

        LIB_LLAISYS.qwen2ResetKV(self._model)

        generated = list(inputs)

        # Prefill: create input_ids and logits on the model's device (NVIDIA)
        input_ids_np = np.array(inputs, dtype=np.int64)
        input_tensor = _create_tensor(
            [len(inputs)], DataType.I64, self.device, self.device_id
        )
        _tensor_load(input_tensor, input_ids_np)

        logits_tensor = _create_tensor(
            [vocab_size], DataType.F32, self.device, self.device_id
        )

        LIB_LLAISYS.qwen2Forward(self._model, input_tensor, logits_tensor)

        # Copy logits from GPU to CPU for sampling
        from ..runtime import RuntimeAPI
        api = RuntimeAPI(self.device)
        logits_np = np.zeros(vocab_size, dtype=np.float32)
        api.memcpy_sync(
            logits_np.ctypes.data,
            _tensor_get_data(logits_tensor),
            vocab_size * 4,
            MemcpyKind.D2H,
        )
        api.device_synchronize()

        EOS_TOKEN_ID = 151643
        next_token = self._sample(logits_np, temperature, top_k, top_p)
        generated.append(next_token)
        _destroy_tensor(input_tensor)
        _destroy_tensor(logits_tensor)

        for _ in range(1, max_new_tokens):
            if next_token == EOS_TOKEN_ID:
                break
            input_ids_np = np.array([next_token], dtype=np.int64)
            input_tensor = _create_tensor(
                [1], DataType.I64, self.device, self.device_id
            )
            _tensor_load(input_tensor, input_ids_np)

            logits_tensor = _create_tensor(
                [vocab_size], DataType.F32, self.device, self.device_id
            )

            LIB_LLAISYS.qwen2Forward(self._model, input_tensor, logits_tensor)

            logits_np = np.zeros(vocab_size, dtype=np.float32)
            api.memcpy_sync(
                logits_np.ctypes.data,
                _tensor_get_data(logits_tensor),
                vocab_size * 4,
                MemcpyKind.D2H,
            )
            api.device_synchronize()

            next_token = self._sample(logits_np, temperature, top_k, top_p)
            generated.append(next_token)
            _destroy_tensor(input_tensor)
            _destroy_tensor(logits_tensor)

        return generated

    def _sample(self, logits, temperature, top_k, top_p):
        if temperature > 0:
            logits = logits / temperature
        if top_k > 1:
            return self._sample_top_k(logits, top_k)
        elif top_p < 1.0:
            return self._sample_top_p(logits, top_p)
        else:
            return int(np.argmax(logits))

    def _sample_top_k(self, logits: np.ndarray, k: int) -> int:
        k = min(max(k, 1), len(logits))
        indices = np.argpartition(logits, -k)[-k:]
        top_k_logits = logits[indices]
        top_k_logits = top_k_logits - np.max(top_k_logits)
        probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
        return int(indices[np.random.choice(len(indices), p=probs)])

    def _sample_top_p(self, logits: np.ndarray, p: float) -> int:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        cumsum = np.cumsum(np.exp(sorted_logits - np.max(sorted_logits)))
        cumsum = cumsum / cumsum[-1]
        cutoff_idx = int(np.searchsorted(cumsum, p)) + 1
        top_p_indices = sorted_indices[:cutoff_idx]
        top_p_logits = logits[top_p_indices]
        top_p_logits = top_p_logits - np.max(top_p_logits)
        probs = np.exp(top_p_logits) / np.sum(np.exp(top_p_logits))
        return int(top_p_indices[np.random.choice(len(top_p_indices), p=probs)])

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.qwen2Destroy(self._model)