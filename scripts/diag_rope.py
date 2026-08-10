"""Diagnose RoPE mismatch between LLAISYS and Torch."""
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "test"))
import llaisys
import torch
from test_utils import arrange_tensor, random_tensor, check_equal


def torch_rope(y: torch.Tensor, x: torch.Tensor, pos_ids: torch.Tensor, theta: float):
    assert y.dim() == 3
    seq_len, n_heads, head_dim = y.shape
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE."

    x_a, x_b = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    positions = pos_ids.to(torch.float32).unsqueeze(1)  # [seq_len, 1]

    i = torch.arange(0, head_dim // 2, dtype=torch.float32, device=y.device)
    freqs = positions / (theta ** (2 * i / head_dim))

    sin, cos = freqs.sin(), freqs.cos()
    sin = sin.unsqueeze(1)
    cos = cos.unsqueeze(1)

    y[..., : head_dim // 2] = x_a * cos - x_b * sin
    y[..., head_dim // 2 :] = x_b * cos + x_a * sin


def main():
    device_name = "nvidia"
    shape = (512, 4, 4096)
    start_end = (512, 1024)
    dtype_name = "f32"
    theta = 10000.0

    print(f"Diagnosing RoPE: shape={shape}, range={start_end}, dtype={dtype_name}")

    x, x_ = random_tensor(shape, dtype_name, device_name)
    pos_ids, pos_ids_ = arrange_tensor(start_end[0], start_end[1], device_name)
    y, y_ = random_tensor(shape, dtype_name, device_name)

    torch_rope(y, x, pos_ids, theta)
    llaisys.Ops.rope(y_, x_, pos_ids_, theta)

    # Copy LLAISYS result to torch tensor for comparison
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    result = torch.zeros(shape, dtype=torch.float32, device="cuda:0")
    api.memcpy_sync(
        result.data_ptr(),
        y_.data_ptr(),
        result.numel() * result.element_size(),
        llaisys.MemcpyKind.D2D,
    )

    diff = (result - y).abs()
    print(f"Max abs diff: {diff.max().item()}")
    print(f"Mean abs diff: {diff.mean().item()}")

    # Check allclose with different tolerances
    for atol, rtol in [(1e-4, 1e-4), (1e-3, 1e-3), (1e-2, 1e-2), (1e-1, 1e-1)]:
        ok = torch.allclose(result, y, atol=atol, rtol=rtol)
        print(f"  allclose(atol={atol}, rtol={rtol}): {ok}")

    # Find where the max diff occurs
    max_idx = diff.argmax()
    flat_idx = max_idx.item()
    s = flat_idx // (4 * 4096)
    h = (flat_idx // 4096) % 4
    d = flat_idx % 4096
    print(f"Max diff at [seq={s}, head={h}, dim={d}]")
    print(f"  LLAISYS: {result[s, h, d].item()}")
    print(f"  Torch:   {y[s, h, d].item()}")
    print(f"  diff:    {diff[s, h, d].item()}")

    # Check the cos/sin computation
    pos = float(pos_ids[s].item())
    half_dim = 4096 // 2
    j = d if d < half_dim else d - half_dim
    freq = pos / (theta ** (2.0 * j / 4096))
    print(f"\n  pos={pos}, j={j}, freq={freq}")
    print(f"  cos(freq)={torch.cos(torch.tensor(freq)).item()}")
    print(f"  sin(freq)={torch.sin(torch.tensor(freq)).item()}")

    # Check if the issue is in the cos/sin computation
    # Compare C++ cos/sin vs torch cos/sin
    i_torch = torch.arange(0, half_dim, dtype=torch.float32, device="cuda:0")
    freqs_torch = torch.tensor(pos, dtype=torch.float32, device="cuda:0") / (theta ** (2 * i_torch / 4096))
    cos_torch = freqs_torch.cos()
    sin_torch = freqs_torch.sin()

    # Compute C++ style cos/sin
    import math
    cos_cpp = []
    sin_cpp = []
    for jj in range(half_dim):
        f = pos / math.pow(theta, 2.0 * jj / 4096)
        cos_cpp.append(math.cos(f))
        sin_cpp.append(math.sin(f))
    cos_cpp = torch.tensor(cos_cpp, dtype=torch.float32, device="cuda:0")
    sin_cpp = torch.tensor(sin_cpp, dtype=torch.float32, device="cuda:0")

    cos_diff = (cos_cpp - cos_torch).abs().max().item()
    sin_diff = (sin_cpp - sin_torch).abs().max().item()
    print(f"\nMax cos diff (C++ vs Torch): {cos_diff}")
    print(f"Max sin diff (C++ vs Torch): {sin_diff}")

    # Check if the issue is in the kernel itself
    # Compute expected output using C++ cos/sin
    x_s = x[s].cpu().numpy()
    x_a = x_s[:half_dim]
    x_b = x_s[half_dim:]
    cos_np = cos_cpp.cpu().numpy()
    sin_np = sin_cpp.cpu().numpy()
    out_first = x_a * cos_np - x_b * sin_np
    out_second = x_b * cos_np + x_a * sin_np
    expected = torch.tensor(
        list(out_first) + list(out_second),
        dtype=torch.float32,
        device="cuda:0",
    )
    actual = result[s, h]
    exp_diff = (actual - expected).abs().max().item()
    print(f"\nMax diff (kernel vs C++ cos/sin expected): {exp_diff}")


if __name__ == "__main__":
    main()
