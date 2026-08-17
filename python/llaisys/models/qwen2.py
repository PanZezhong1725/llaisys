from ctypes import byref, c_int, c_int64, c_size_t, c_void_p
import json
import ml_dtypes  # 为 NumPy 注册 bfloat16 dtype
from safetensors import safe_open
from pathlib import Path
from typing import Sequence

from ..libllaisys import DataType
from ..libllaisys import DeviceType
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import LlaisysQwen2Meta


_DTYPE_MAP = {
    "bfloat16": DataType.BF16,
    "float16": DataType.F16,
    "float32": DataType.F32,
}


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self._model = None
        self._weight_map = {}

        self._model_path = Path(model_path).resolve()
        config_path = self._model_path / "config.json"

        if not self._model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self._model_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        dtype_name = config["torch_dtype"]
        if dtype_name not in _DTYPE_MAP:
            raise ValueError(f"Unsupported model dtype: {dtype_name}")

        head_dim = config["hidden_size"] // config["num_attention_heads"]
        end_token = config["eos_token_id"]
        if isinstance(end_token, list):
            end_token = end_token[0]

        self._meta = LlaisysQwen2Meta(
            dtype=_DTYPE_MAP[dtype_name],
            nlayer=config["num_hidden_layers"],
            hs=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config["num_key_value_heads"],
            dh=head_dim,
            di=config["intermediate_size"],
            maxseq=config["max_position_embeddings"],
            voc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"],
            end_token=end_token,
        )

        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self._meta),
            DeviceType(device),
            device_ids,
            1,
        )
        if not self._model:
            raise RuntimeError("Failed to create the Qwen2 backend model.")

        weights_pointer = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        if not weights_pointer:
            raise RuntimeError("Failed to get the Qwen2 weight handles.")

        self._weight_map = self._build_weight_map(weights_pointer.contents)
        self._loaded_weight_count = self._load_weights()

    def __del__(self):
        model = getattr(self, "_model", None)
        if model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
            self._model = None

    def _build_weight_map(self, weights):
        result = {
            "model.embed_tokens.weight": weights.in_embed,
            "model.norm.weight": weights.out_norm_w,
            "lm_head.weight": weights.out_embed,
        }

        layer_fields = {
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

        for layer_index in range(self._meta.nlayer):
            for suffix, field_name in layer_fields.items():
                name = f"model.layers.{layer_index}.{suffix}"
                result[name] = getattr(weights, field_name)[layer_index]

        expected = 3 + 12 * self._meta.nlayer
        if len(result) != expected or not all(result.values()):
            raise RuntimeError("Qwen2 weight handle map is incomplete.")

        return result

    @staticmethod
    def _tensor_shape(tensor):
        ndim = LIB_LLAISYS.tensorGetNdim(tensor)
        shape_buffer = (c_size_t * ndim)()
        LIB_LLAISYS.tensorGetShape(tensor, shape_buffer)
        return tuple(shape_buffer)

    def _load_weights(self):
        weight_files = sorted(self._model_path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(
                f"No safetensors files found in: {self._model_path}"
            )

        dtype_names = {
            DataType.BF16: "bfloat16",
            DataType.F16: "float16",
            DataType.F32: "float32",
        }
        loaded = set()

        for weight_file in weight_files:
            print(f"Loading weights from {weight_file.name}")

            with safe_open(
                str(weight_file),
                framework="numpy",
                device="cpu",
            ) as safetensor_file:
                for name in safetensor_file.keys():
                    if name not in self._weight_map:
                        raise KeyError(f"Unexpected model weight: {name}")
                    if name in loaded:
                        raise RuntimeError(f"Duplicate model weight: {name}")

                    tensor = self._weight_map[name]
                    array = safetensor_file.get_tensor(name)

                    expected_shape = self._tensor_shape(tensor)
                    if tuple(array.shape) != expected_shape:
                        raise ValueError(
                            f"Weight shape mismatch for {name}: "
                            f"expected {expected_shape}, got {array.shape}"
                        )

                    backend_dtype = DataType(LIB_LLAISYS.tensorGetDataType(tensor))
                    expected_dtype = dtype_names[backend_dtype]
                    if str(array.dtype) != expected_dtype:
                        raise ValueError(
                            f"Weight dtype mismatch for {name}: "
                            f"expected {expected_dtype}, got {array.dtype}"
                        )

                    if not array.flags.c_contiguous:
                        raise ValueError(f"Weight is not contiguous: {name}")

                    LIB_LLAISYS.tensorLoad(
                        tensor,
                        c_void_p(array.ctypes.data),
                    )
                    loaded.add(name)

        missing = set(self._weight_map) - loaded
        if missing:
            examples = ", ".join(sorted(missing)[:3])
            raise RuntimeError(
                f"Missing {len(missing)} model weights, examples: {examples}"
            )

        print(f"Loaded {len(loaded)} model weights")
        return len(loaded)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # 作业要求使用 argmax，因此这三个采样参数暂不参与计算。
        _ = top_k, top_p, temperature

        input_tokens = [int(token) for token in inputs]
        if not input_tokens:
            raise ValueError("inputs must contain at least one token.")

        if any(token < 0 or token >= self._meta.voc for token in input_tokens):
            raise ValueError("inputs contain an invalid token id.")

        if max_new_tokens is None:
            max_new_tokens = 128

        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer.")

        if len(input_tokens) + max_new_tokens > self._meta.maxseq:
            raise ValueError(
                "input and generated tokens exceed maximum sequence length."
            )

        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)

        output_tokens = input_tokens.copy()
        current_tokens = input_tokens

        for _ in range(max_new_tokens):
            token_buffer = (c_int64 * len(current_tokens))(*current_tokens)

            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                token_buffer,
                len(current_tokens),
            )

            if next_token < 0:
                raise RuntimeError("Qwen2 backend inference failed.")

            output_tokens.append(next_token)

            if next_token == self._meta.end_token:
                break

            current_tokens = [next_token]

        return output_tokens
