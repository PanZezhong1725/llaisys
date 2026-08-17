import argparse
from ctypes import c_float, c_ubyte, c_void_p, cast, memset, sizeof, string_at

import llaisys


def test_basic_runtime_api(device_name: str = "cpu"):
    device_type = (
        llaisys.DeviceType.CPU
        if device_name == "cpu"
        else llaisys.DeviceType.NVIDIA
    )
    api = llaisys.RuntimeAPI(device_type)

    ndev = api.get_device_count()
    print(f"Found {ndev} {device_name} devices")
    if ndev == 0:
        print("     Skipped")
        return

    for i in range(ndev):
        print(f"Testing device {i}...")
        api.set_device(i)
        test_memcpy(api, 1024 * 1024)
        test_async_memcpy(api, 1024 * 1024)

        print("     Passed")

    test_invalid_device_ids(api, ndev)
    if device_name == "nvidia":
        test_context_device_switching(ndev)


def test_memcpy(api, size_bytes: int):
    host_a = (c_ubyte * size_bytes)()
    host_b = (c_ubyte * size_bytes)()
    memset(host_a, 0xA5, size_bytes)
    memset(host_b, 0, size_bytes)
    device_a = api.malloc_device(size_bytes)
    device_b = api.malloc_device(size_bytes)
    try:
        # a -> device_a
        api.memcpy_sync(
            device_a,
            cast(host_a, c_void_p),
            size_bytes,
            llaisys.MemcpyKind.H2D,
        )
        # device_a -> device_b
        api.memcpy_sync(
            device_b,
            device_a,
            size_bytes,
            llaisys.MemcpyKind.D2D,
        )
        # device_b -> b
        api.memcpy_sync(
            cast(host_b, c_void_p),
            device_b,
            size_bytes,
            llaisys.MemcpyKind.D2H,
        )

        assert bytes(host_a) == bytes(host_b)
    finally:
        api.free_device(device_b)
        api.free_device(device_a)


def test_async_memcpy(api, size_bytes: int):
    host_a = api.malloc_host(size_bytes)
    host_b = api.malloc_host(size_bytes)
    device_a = api.malloc_device(size_bytes)
    device_b = api.malloc_device(size_bytes)
    stream = api.create_stream()
    try:
        memset(host_a, 0x5A, size_bytes)
        memset(host_b, 0, size_bytes)
        api.memcpy_async(
            device_a, host_a, size_bytes, llaisys.MemcpyKind.H2D, stream
        )
        api.memcpy_async(
            device_b, device_a, size_bytes, llaisys.MemcpyKind.D2D, stream
        )
        api.memcpy_async(
            host_b, device_b, size_bytes, llaisys.MemcpyKind.D2H, stream
        )
        api.stream_synchronize(stream)
        assert string_at(host_b, size_bytes) == bytes([0x5A]) * size_bytes
    finally:
        api.destroy_stream(stream)
        api.free_device(device_b)
        api.free_device(device_a)
        api.free_host(host_b)
        api.free_host(host_a)


def test_invalid_device_ids(api, device_count: int):
    for device_id in (-1, device_count):
        try:
            api.set_device(device_id)
        except ValueError:
            continue
        raise AssertionError(f"invalid device id {device_id} was accepted")


def test_context_device_switching(device_count: int):
    if device_count < 2:
        print("     Multi-GPU context test skipped (requires at least 2 devices)")
        return

    host_a = (c_float * 4)(1.0, 2.0, 3.0, 4.0)
    host_b = (c_float * 4)(5.0, 6.0, 7.0, 8.0)
    expected = [6.0, 8.0, 10.0, 12.0]
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)

    for _ in range(2):
        for device_id in range(device_count):
            a = llaisys.Tensor(
                (4,), llaisys.DataType.F32, llaisys.DeviceType.NVIDIA, device_id
            )
            b = llaisys.Tensor(
                (4,), llaisys.DataType.F32, llaisys.DeviceType.NVIDIA, device_id
            )
            out = llaisys.Tensor(
                (4,), llaisys.DataType.F32, llaisys.DeviceType.NVIDIA, device_id
            )
            a.load(cast(host_a, c_void_p))
            b.load(cast(host_b, c_void_p))
            llaisys.Ops.add(out, a, b)

            host_out = (c_float * 4)()
            api.set_device(device_id)
            api.memcpy_sync(
                cast(host_out, c_void_p),
                out.data_ptr(),
                sizeof(host_out),
                llaisys.MemcpyKind.D2H,
            )
            assert list(host_out) == expected
            del out, b, a

    print("     Multi-GPU context switching passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    args = parser.parse_args()
    test_basic_runtime_api(args.device)

    print("\033[92mTest passed!\033[0m\n")
