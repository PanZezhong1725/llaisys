from __future__ import annotations

import json
import logging
import re
from ctypes import byref, c_int, c_int64, c_size_t, c_void_p
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping, Sequence

import numpy as np
import safetensors

from ..libllaisys import LIB_LLAISYS, DataType, DeviceType
from ..libllaisys.qwen2 import LlaisysQwen2Meta


logger = logging.getLogger(__name__)

_DEFAULT_MAX_NEW_TOKENS = 128
_DEVICE_COUNT = 1

GLOBAL_WEIGHT_MAPPING: dict[str, str] = {
    "model.embed_tokens.weight": "in_embed",
    "model.norm.weight": "out_norm_w",
    "lm_head.weight": "out_embed",
}

LAYER_WEIGHT_MAPPING: dict[str, str] = {
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

# 解析（层编号，权重参数名称）
_LAYER_PATTERN = re.compile(r"model\.layers\.(\d+)\.(.+)")


def resolve_weight(weights: Any, name: str) -> Any:
    """返回 safetensors 权重名称对应的后端 tensor handle"""
    global_field = GLOBAL_WEIGHT_MAPPING.get(name)
    if global_field is not None:
        return getattr(weights, global_field)

    match = _LAYER_PATTERN.fullmatch(name)
    if match is None:
        raise KeyError(f"Unknown Qwen2 weight: {name}")

    layer_index = int(match.group(1))
    layer_weight_name = match.group(2)

    layer_field = LAYER_WEIGHT_MAPPING.get(layer_weight_name)
    if layer_field is None:
        raise KeyError(f"Unknown layer weight: {name}")

    tensor_array = getattr(weights, layer_field)

    try:
        return tensor_array[layer_index]
    except IndexError as exc:
        raise IndexError(
            f"Layer index {layer_index} is out of range for {name}"
        ) from exc


def config_dtype_to_llaisys(
    config: Mapping[str, Any],
) -> DataType:
    """将模型配置中的 dtype 转换为 LLAISYS 数据类型"""
    raw_dtype = (
        config.get("torch_dtype")
        or config.get("dtype")
        or "bfloat16"
    )

    dtype = str(raw_dtype).removeprefix("torch.")

    mapping = {
        "bfloat16": DataType.BF16,
        "float16": DataType.F16,
        "float32": DataType.F32,
    }

    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model dtype: {raw_dtype}"
        ) from exc


class Qwen2:
    """基于 LLAISYS 后端的 Qwen2 模型封装"""

    def __init__(
        self,
        model_path: str | Path,
        device: DeviceType = DeviceType.CPU,
    ) -> None:
        # 保留库对象引用，避免解释器退出时模块全局变量先被清理。
        self._lib = LIB_LLAISYS
        self._model: Any | None = None
        self._weights: Any | None = None

        model_path = Path(model_path)
        config_path = model_path / "config.json"

        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        if not isinstance(config, dict):
            raise TypeError(
                f"Expected an object in {config_path}, "
                f"got {type(config).__name__}"
            )

        self._config: dict[str, Any] = config
        self._device = device

        hidden_size = int(config["hidden_size"])
        num_heads = int(config["num_attention_heads"])

        if num_heads <= 0:
            raise ValueError(
                "num_attention_heads must be greater than zero"
            )

        configured_head_dim = config.get("head_dim")
        if configured_head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError(
                    "hidden_size must be divisible by "
                    "num_attention_heads when head_dim is absent"
                )

            head_dim = hidden_size // num_heads
        else:
            head_dim = int(configured_head_dim)

        eos_token_id = config["eos_token_id"]
        if not isinstance(eos_token_id, int):
            raise TypeError(
                "This backend requires eos_token_id to be an integer"
            )

        self._eos_token_id = eos_token_id

        meta = LlaisysQwen2Meta(
            dtype=int(config_dtype_to_llaisys(config)),
            nlayer=int(config["num_hidden_layers"]),
            hs=hidden_size,
            nh=num_heads,
            nkvh=int(config["num_key_value_heads"]),
            dh=head_dim,
            di=int(config["intermediate_size"]),
            maxseq=int(config["max_position_embeddings"]),
            voc=int(config["vocab_size"]),
            epsilon=float(config["rms_norm_eps"]),
            theta=float(config["rope_theta"]),
            end_token=self._eos_token_id,
        )

        # 当前后端只使用单个设备
        device_ids = (c_int * _DEVICE_COUNT)(0)

        # 在后端创建权重 tensor，未加载权重 ，返回 LlaisysQwen2Model 指针
        model = self._lib.llaisysQwen2ModelCreate(
            byref(meta),
            int(device),
            device_ids,
            _DEVICE_COUNT,
        )
        if not model:
            raise RuntimeError(
                "llaisysQwen2ModelCreate returned null"
            )

        self._model = model

        try:
            weights_pointer = (
                self._lib.llaisysQwen2ModelWeights(self._model)
            )
            if not weights_pointer:
                raise RuntimeError(
                    "llaisysQwen2ModelWeights returned null"
                )

            self._weights = weights_pointer.contents
            self._load_model_weights(model_path)
        except Exception:
            try:
                self.close()
            except Exception:
                logger.exception(
                    "Failed to release Qwen2 model after initialization error"
                )
            raise

    def __del__(self) -> None:
        """尽力释放仍未关闭的后端资源"""
        try:
            self.close()
        except Exception:
            # 析构函数不能向外传播异常。
            pass

    def __enter__(self) -> Qwen2:
        """返回可在 with 语句中使用的模型实例"""
        if self._model is None:
            raise RuntimeError(
                "Cannot enter context: model has been closed"
            )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """退出上下文时释放后端资源"""
        self.close()

    def close(self) -> None:
        """关闭模型并释放后端资源"""
        model = getattr(self, "_model", None)
        if model is None:
            return

        # 先清空句柄，避免销毁函数抛出异常后发生重复释放。
        self._model = None
        self._weights = None

        lib = getattr(self, "_lib", None)
        if lib is not None:
            lib.llaisysQwen2ModelDestroy(model)


    def _load_model_weights(
        self,
        model_path: Path,
    ) -> None:
        """从模型目录加载所有 safetensors 权重分片"""
        # 注册 NumPy 的 bfloat16 dtype
        import ml_dtypes  # noqa: F401

        if self._weights is None:
            raise RuntimeError("Model weights are not initialized")

        shard_paths = sorted(model_path.glob("*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")

        loaded_names: set[str] = set()

        for shard_path in shard_paths:
            logger.info("Loading weight shard: %s", shard_path.name)

            # 使用上下文管理器及时关闭内存映射
            with safetensors.safe_open(str(shard_path), framework="numpy", device="cpu") as data:
                for name in data.keys():
                    if name in loaded_names:
                        raise RuntimeError(f"Duplicate tensor: {name}")

                    source = data.get_tensor(name)
                    target = resolve_weight(self._weights, name)

                    logger.debug(
                        "Loading tensor %s: shape=%s, dtype=%s",
                        name,
                        source.shape,
                        source.dtype,
                    )

                    self._load_tensor(target, source, name)
                    loaded_names.add(name)

        logger.info(
            "Loaded %d tensors from %d shards",
            len(loaded_names),
            len(shard_paths),
        )



    def _load_tensor(
        self,
        target: Any,
        source: np.ndarray,
        name: str,
    ) -> None:
        """将一个 NumPy 张量加载到后端 tensor 中"""
        if not target:
            raise RuntimeError(
                f"Backend tensor is null: {name}"
            )

        # tensorLoad 接收裸指针，源数据必须连续。
        source = np.ascontiguousarray(source)

        ndim = int(self._lib.tensorGetNdim(target))
        if ndim < 0:
            raise RuntimeError(
                f"Invalid tensor dimension for {name}: {ndim}"
            )

        shape_buffer = (c_size_t * ndim)()
        self._lib.tensorGetShape(
            target,
            shape_buffer,
        )

        target_shape = tuple(
            int(dimension)
            for dimension in shape_buffer
        )
        source_shape = tuple(source.shape)

        if target_shape != source_shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"safetensors={source_shape}, "
                f"backend={target_shape}"
            )

        source_pointer = c_void_p(int(source.ctypes.data))
        self._lib.tensorLoad(target, source_pointer)


    def _check_token(self, token: int):
        vocab_size = int(self._config["vocab_size"])

        if token < 0 or token >= vocab_size:
            raise RuntimeError(f"Invalid token returned by backend: {token}")


    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        del top_p
        del temperature

        tokens = [int(token) for token in inputs]

        if not tokens:
            raise ValueError("inputs must not be empty")

        if max_new_tokens is None:
            max_new_tokens = 128

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        if top_k != 1:
            raise NotImplementedError("Curent LLAISYS backend only support argmax")

        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)

        # 第一次：完整 Prompt。
        prompt_array = (c_int64 * len(tokens))(*tokens)

        next_token = int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                prompt_array,
                len(tokens),
            )
        )

        self._check_token(next_token)
        tokens.append(next_token)

        # 后续：每次只传刚生成的一个 token。
        for _ in range(max_new_tokens - 1):
            if tokens[-1] == self._eos_token_id:
                break

            one_token = (c_int64 * 1)(tokens[-1])

            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model,
                    one_token,
                    1,
                )
            )

            self._check_token(next_token)
            tokens.append(next_token)

        return tokens