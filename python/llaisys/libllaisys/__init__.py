import os
import sys
import ctypes
from pathlib import Path

from .runtime import load_runtime
from .runtime import LlaisysRuntimeAPI
from .llaisys_types import llaisysDeviceType_t, DeviceType
from .llaisys_types import llaisysDataType_t, DataType
from .llaisys_types import llaisysMemcpyKind_t, MemcpyKind
from .llaisys_types import llaisysStream_t
from .tensor import llaisysTensor_t
from .tensor import load_tensor
from .ops import load_ops


def load_shared_library():
    lib_dir = Path(__file__).parent

    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "llaisys.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    lib_path = os.path.join(lib_dir, libname)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Shared library not found: {lib_path}")

    # CUDA kernels are built as a separate shared library so nvcc can perform
    # device linking. Load it globally before the main C API library.
    if sys.platform.startswith("linux"):
        corex_home = os.environ.get("COREX_HOME", "/usr/local/corex")
        corex_cudart = Path(corex_home) / "lib64" / "libcudart.so"
        if corex_cudart.is_file():
            ctypes.CDLL(str(corex_cudart), mode=ctypes.RTLD_GLOBAL)
        cuda_lib = lib_dir / "libllaisys-ops-nvidia.so"
        if cuda_lib.is_file():
            ctypes.CDLL(str(cuda_lib), mode=ctypes.RTLD_GLOBAL)
        else:
            # CoreX links GPU kernels into the main library as a static archive.
            pass

    return ctypes.CDLL(str(lib_path))


LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)


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
    "llaisysStream_t",
]
