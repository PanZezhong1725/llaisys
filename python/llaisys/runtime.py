from ctypes import c_void_p

from . import libllaisys
from .libllaisys import LIB_LLAISYS


class RuntimeAPI:
    """Thin Python facade over a device runtime function table."""

    def __init__(self, device_type: libllaisys.DeviceType):
        self.device_type = libllaisys.DeviceType(device_type)
        self._table = LIB_LLAISYS.llaisysGetRuntimeAPI(
            libllaisys.llaisysDeviceType_t(self.device_type)
        )
        if not self._table:
            raise RuntimeError(f"Runtime is unavailable for {self.device_type!r}")

    def get_device_count(self) -> int:
        return int(self._table.contents.get_device_count())

    def set_device(self, device_id: int) -> None:
        count = self.get_device_count()
        if not 0 <= device_id < count:
            raise ValueError(f"device_id must be in [0, {count}), got {device_id}")
        self._table.contents.set_device(device_id)

    def device_synchronize(self) -> None:
        self._table.contents.device_synchronize()

    def create_stream(self) -> libllaisys.llaisysStream_t:
        return self._table.contents.create_stream()

    def destroy_stream(self, stream: libllaisys.llaisysStream_t) -> None:
        self._table.contents.destroy_stream(stream)

    def stream_synchronize(self, stream: libllaisys.llaisysStream_t) -> None:
        self._table.contents.stream_synchronize(stream)

    def malloc_device(self, size: int) -> c_void_p:
        return self._table.contents.malloc_device(size)

    def free_device(self, pointer: c_void_p) -> None:
        self._table.contents.free_device(pointer)

    def malloc_host(self, size: int) -> c_void_p:
        return self._table.contents.malloc_host(size)

    def free_host(self, pointer: c_void_p) -> None:
        self._table.contents.free_host(pointer)

    def memcpy_sync(self, destination, source, size: int, kind: libllaisys.MemcpyKind) -> None:
        self._table.contents.memcpy_sync(
            destination,
            source,
            size,
            libllaisys.llaisysMemcpyKind_t(kind),
        )

    def memcpy_async(
        self,
        destination,
        source,
        size: int,
        kind: libllaisys.MemcpyKind,
        stream: libllaisys.llaisysStream_t,
    ) -> None:
        self._table.contents.memcpy_async(
            destination,
            source,
            size,
            libllaisys.llaisysMemcpyKind_t(kind),
            stream,
        )
