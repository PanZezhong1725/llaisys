from ctypes import POINTER, c_uint8, c_void_p, c_size_t, c_ssize_t, c_int
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t

# Handle type
llaisysTensor_t = c_void_p


def load_tensor(lib):
    lib.tensorCreate.argtypes = [
        POINTER(c_size_t),  # shape
        c_size_t,  # ndim
        llaisysDataType_t,  # dtype
        llaisysDeviceType_t,  # device_type
        c_int,  # device_id
    ]
    lib.tensorCreate.restype = llaisysTensor_t

    lib.tensorDestroy.argtypes = [llaisysTensor_t]
    lib.tensorDestroy.restype = None

    lib.tensorGetData.argtypes = [llaisysTensor_t]
    lib.tensorGetData.restype = c_void_p

    lib.tensorGetNdim.argtypes = [llaisysTensor_t]
    lib.tensorGetNdim.restype = c_size_t

    lib.tensorGetShape.argtypes = [llaisysTensor_t, POINTER(c_size_t)]
    lib.tensorGetShape.restype = None

    lib.tensorGetStrides.argtypes = [llaisysTensor_t, POINTER(c_ssize_t)]
    lib.tensorGetStrides.restype = None

    lib.tensorGetDataType.argtypes = [llaisysTensor_t]
    lib.tensorGetDataType.restype = llaisysDataType_t

    lib.tensorGetDeviceType.argtypes = [llaisysTensor_t]
    lib.tensorGetDeviceType.restype = llaisysDeviceType_t

    lib.tensorGetDeviceId.argtypes = [llaisysTensor_t]
    lib.tensorGetDeviceId.restype = c_int

    lib.tensorDebug.argtypes = [llaisysTensor_t]
    lib.tensorDebug.restype = None

    lib.tensorIsContiguous.argtypes = [llaisysTensor_t]
    lib.tensorIsContiguous.restype = c_uint8

    lib.tensorLoad.argtypes = [llaisysTensor_t, c_void_p]
    lib.tensorLoad.restype = None

    lib.tensorView.argtypes = [llaisysTensor_t, POINTER(c_size_t), c_size_t]
    lib.tensorView.restype = llaisysTensor_t

    lib.tensorPermute.argtypes = [llaisysTensor_t, POINTER(c_size_t)]
    lib.tensorPermute.restype = llaisysTensor_t

    lib.tensorSlice.argtypes = [
        llaisysTensor_t,  # tensor handle
        c_size_t,  # dim  : which axis to slice
        c_size_t,  # start: inclusive
        c_size_t,  # end  : exclusive
    ]
    lib.tensorSlice.restype = llaisysTensor_t

    lib.tensorReshape.argtypes = [llaisysTensor_t, POINTER(c_size_t), c_size_t]
    lib.tensorReshape.restype = llaisysTensor_t
