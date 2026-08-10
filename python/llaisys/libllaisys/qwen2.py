from ctypes import c_void_p, c_int, c_char_p
from .tensor import llaisysTensor_t
from .llaisys_types import llaisysDeviceType_t

# Handle type for Qwen2 model
llaisysQwen2Model_t = c_void_p


def load_qwen2(lib):
    # qwen2Create
    lib.qwen2Create.argtypes = [llaisysDeviceType_t, c_int]
    lib.qwen2Create.restype = llaisysQwen2Model_t

    # qwen2Destroy
    lib.qwen2Destroy.argtypes = [llaisysQwen2Model_t]
    lib.qwen2Destroy.restype = None

    # qwen2LoadWeight
    lib.qwen2LoadWeight.argtypes = [
        llaisysQwen2Model_t,
        c_char_p,
        llaisysTensor_t,
    ]
    lib.qwen2LoadWeight.restype = None

    # qwen2Forward
    lib.qwen2Forward.argtypes = [
        llaisysQwen2Model_t,
        llaisysTensor_t,
        llaisysTensor_t,
    ]
    lib.qwen2Forward.restype = None

    # qwen2ResetKV
    lib.qwen2ResetKV.argtypes = [llaisysQwen2Model_t]
    lib.qwen2ResetKV.restype = None
