import ctypes 
from ctypes import c_size_t, c_int, c_int64,POINTER
from .llaisys_types import llaisysDataType_t,llaisysDeviceType_t
from .tensor import llaisysTensor_t
class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]

LlaisysQwen2Model = ctypes.c_void_p 

class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed",llaisysTensor_t),
        ("out_embed",llaisysTensor_t),
        ("out_norm_w",llaisysTensor_t),
        ("attn_norm_w",POINTER(llaisysTensor_t)),
        ("attn_q_w",POINTER(llaisysTensor_t)),
        ("attn_q_b",POINTER(llaisysTensor_t)),
        ("attn_k_w",POINTER(llaisysTensor_t)),
        ("attn_k_b",POINTER(llaisysTensor_t)),
        ("attn_v_w",POINTER(llaisysTensor_t)),
        ("attn_v_b",POINTER(llaisysTensor_t)),
        ("attn_o_w",POINTER(llaisysTensor_t)),
        ("mlp_norm_w",POINTER(llaisysTensor_t)),
        ("mlp_gate_w",POINTER(llaisysTensor_t)),
        ("mlp_up_w",POINTER(llaisysTensor_t)),
        ("mlp_down_w",POINTER(llaisysTensor_t)),
    ]
def load_model(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [POINTER(LlaisysQwen2Meta), llaisysDeviceType_t, POINTER(c_int), c_int]
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model

    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelDestroy.restype  = None   

    lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)    

    lib.llaisysQwen2ModelInfer.argtypes = [LlaisysQwen2Model, POINTER(c_int64) , c_size_t ]
    lib.llaisysQwen2ModelInfer.restype  = c_int64

