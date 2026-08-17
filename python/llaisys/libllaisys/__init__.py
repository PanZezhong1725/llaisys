import ctypes
import sys
from pathlib import Path

from .llaisys_types import (
    DataType,
    DeviceType,
    MemcpyKind,
    llaisysDataType_t,
    llaisysDeviceType_t,
    llaisysMemcpyKind_t,
    llaisysStream_t,
)
from .ops import load_ops
from .qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights, llaisysQwen2Model_t, load_qwen2
from .runtime import LlaisysRuntimeAPI, load_runtime
from .tensor import llaisysTensor_t, load_tensor


def _library_candidates():
    if sys.platform.startswith("linux"):
        return ("libllaisys.so",)
    if sys.platform == "darwin":
        return ("libllaisys.dylib",)
    if sys.platform == "win32":
        # MSVC and MinGW use different default prefixes.
        return ("llaisys.dll", "libllaisys.dll")
    raise RuntimeError(f"Unsupported host platform: {sys.platform}")


def _open_library():
    directory = Path(__file__).resolve().parent
    for filename in _library_candidates():
        candidate = directory / filename
        if candidate.is_file():
            return ctypes.CDLL(str(candidate))
    expected = ", ".join(_library_candidates())
    raise FileNotFoundError(f"Shared library not found in {directory}; expected {expected}")


LIB_LLAISYS = _open_library()
for loader in (load_runtime, load_tensor, load_ops, load_qwen2):
    loader(LIB_LLAISYS)


__all__ = [
    "LIB_LLAISYS",
    "LlaisysRuntimeAPI",
    "llaisysStream_t",
    "llaisysTensor_t",
    "llaisysDataType_t",
    "DataType",
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "LlaisysQwen2Meta",
    "LlaisysQwen2Weights",
    "llaisysQwen2Model_t",
    "load_qwen2",
]
