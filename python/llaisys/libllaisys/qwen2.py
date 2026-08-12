import ctypes
from ctypes import POINTER, Structure, c_char_p, c_float, c_int, c_int64, c_size_t, c_void_p


class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", c_int),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta), c_int, POINTER(c_int), c_int
    ]
    lib.llaisysQwen2ModelCreate.restype = c_void_p
    lib.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
    lib.llaisysQwen2ModelDestroy.restype = None
    lib.llaisysQwen2ModelLoadWeight.argtypes = [
        c_void_p, c_char_p, c_void_p, POINTER(c_size_t), c_size_t, c_int
    ]
    lib.llaisysQwen2ModelLoadWeight.restype = None
    lib.llaisysQwen2ModelReset.argtypes = [c_void_p]
    lib.llaisysQwen2ModelReset.restype = None
    lib.llaisysQwen2ModelInfer.argtypes = [c_void_p, POINTER(c_int64), c_size_t]
    lib.llaisysQwen2ModelInfer.restype = c_int64


__all__ = ["LlaisysQwen2Meta", "load_qwen2"]
