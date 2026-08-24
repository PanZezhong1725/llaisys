"""Diagnose RoPE mismatch for small shapes."""
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "test"))
import llaisys
import torch
from test_utils import arrange_tensor, random_tensor, check_equal


def torch_rope(y, x, pos_ids, theta):
    seq_len, n_heads, head_dim = y.shape
    x_a, x_b = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    positions = pos_ids.to(torch.float32).unsqueeze(1)
    i = torch.arange(0, head_dim // 2, dtype=torch.float32, device=y.device)
    freqs = positions / (theta ** (2 * i / head_dim))
    sin, cos = freqs.sin(), freqs.cos()
    sin = sin.unsqueeze(1)
    cos = cos.unsqueeze(1)
    y[..., : head_dim // 2] = x_a * cos - x_b * sin
    y[..., head_dim // 2 :] = x_b * cos + x_a * sin


def main():
    torch.cuda.synchronize()
    for shape, se in [((2, 1, 4), (0, 2)), ((512, 4, 4096), (512, 1024))]:
        for dn, atol, rtol in [("f32", 1e-4, 1e-4), ("f16", 1e-3, 1e-3), ("bf16", 1e-2, 1e-2)]:
            torch.cuda.synchronize()
            x, x_ = random_tensor(shape, dn, "nvidia")
            pos, pos_ = arrange_tensor(se[0], se[1], "nvidia")
            y, y_ = random_tensor(shape, dn, "nvidia")
            torch_rope(y, x, pos, 10000.0)
            torch.cuda.synchronize()
            llaisys.Ops.rope(y_, x_, pos_, 10000.0)
            torch.cuda.synchronize()

            # Compute max abs diff
            api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
            res = torch.zeros(shape, dtype=torch.float32, device="cuda:0")
            api.memcpy_sync(res.data_ptr(), y_.data_ptr(), res.numel() * res.element_size(), llaisys.MemcpyKind.D2D)
            diff = (res - y).abs().max().item()
            ok = check_equal(y_, y, atol=atol, rtol=rtol)
            print(f"shape={shape} range={se} dtype={dn} atol={atol} maxdiff={diff:.3e} pass={ok}")
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
