from ctypes import c_float

from .libllaisys import LIB_LLAISYS
from .tensor import Tensor


def _handle(tensor):
    return None if tensor is None else tensor.lib_tensor()


class Ops:
    @staticmethod
    def add(output: Tensor, left: Tensor, right: Tensor) -> None:
        LIB_LLAISYS.llaisysAdd(_handle(output), _handle(left), _handle(right))

    @staticmethod
    def argmax(index: Tensor, value: Tensor, input_: Tensor) -> None:
        LIB_LLAISYS.llaisysArgmax(_handle(index), _handle(value), _handle(input_))

    @staticmethod
    def embedding(output: Tensor, indices: Tensor, table: Tensor) -> None:
        LIB_LLAISYS.llaisysEmbedding(_handle(output), _handle(indices), _handle(table))

    @staticmethod
    def linear(output: Tensor, input_: Tensor, weight: Tensor, bias: Tensor = None) -> None:
        LIB_LLAISYS.llaisysLinear(_handle(output), _handle(input_), _handle(weight), _handle(bias))

    @staticmethod
    def rearrange(output: Tensor, input_: Tensor) -> None:
        LIB_LLAISYS.llaisysRearrange(_handle(output), _handle(input_))

    @staticmethod
    def rms_norm(output: Tensor, input_: Tensor, weight: Tensor, epsilon: float) -> None:
        LIB_LLAISYS.llaisysRmsNorm(_handle(output), _handle(input_), _handle(weight), c_float(epsilon))

    @staticmethod
    def rope(output: Tensor, input_: Tensor, positions: Tensor, theta: float) -> None:
        LIB_LLAISYS.llaisysROPE(_handle(output), _handle(input_), _handle(positions), c_float(theta))

    @staticmethod
    def self_attention(output: Tensor, query: Tensor, key: Tensor, value: Tensor, scale: float) -> None:
        LIB_LLAISYS.llaisysSelfAttention(
            _handle(output), _handle(query), _handle(key), _handle(value), c_float(scale)
        )

    @staticmethod
    def swiglu(output: Tensor, gate: Tensor, up: Tensor) -> None:
        LIB_LLAISYS.llaisysSwiGLU(_handle(output), _handle(gate), _handle(up))
