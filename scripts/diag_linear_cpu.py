"""Diagnose CPU linear operator against HF reference."""
import sys, torch, numpy as np, ctypes
sys.path.insert(0, '/data/llaisys-26s/python')
import llaisys
from transformers import AutoModelForCausalLM

model_path = '/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562'

print("Loading HF model...")
hf = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
hf.eval()

# Get first layer q_proj weight [1536, 1536]
w = hf.model.layers[0].self_attn.q_proj.weight.detach().numpy()
print(f"weight shape: {w.shape}, dtype: {w.dtype}")

x = np.random.randn(9, 1536).astype(np.float32)

# HF reference: x @ w.T (linear is in_features=1536, out_features=1536)
ref = x @ w.T
print(f"HF ref shape: {ref.shape}")
print(f"HF ref[0,:5]: {ref[0,:5]}")

# LLAISYS CPU linear
print("\nCreating LLAISYS tensors...")
xt = llaisys.Tensor([9, 1536], llaisys.DataType.F32, llaisys.DeviceType.CPU, 0)
api = llaisys.RuntimeAPI(llaisys.DeviceType.CPU)
api.memcpy_sync(xt.data_ptr(), x.ctypes.data, x.nbytes, llaisys.MemcpyKind.H2D)

wt = llaisys.Tensor([1536, 1536], llaisys.DataType.F32, llaisys.DeviceType.CPU, 0)
api.memcpy_sync(wt.data_ptr(), w.ctypes.data, w.nbytes, llaisys.MemcpyKind.H2D)

out_tensor = llaisys.Tensor([9, 1536], llaisys.DataType.F32, llaisys.DeviceType.CPU, 0)

# Load ops
from llaisys.libllaisys.ops import load_ops
load_ops(llaisys.LIB_LLAISYS)

print("Calling LLAISYS linear...")
llaisys.Ops.linear(out_tensor, xt, wt, None)

out = np.zeros((9, 1536), dtype=np.float32)
api.memcpy_sync(out.ctypes.data, out_tensor.data_ptr(), out.nbytes, llaisys.MemcpyKind.D2H)

print(f"LLAISYS out shape: {out.shape}")
print(f"LLAISYS out[0,:5]: {out[0,:5]}")

diff = np.abs(out - ref)
print(f"\nlinear max diff: {np.max(diff):.6f}")
print(f"linear mean diff: {np.mean(diff):.6f}")
print(f"linear close (atol=1e-3): {np.allclose(out, ref, atol=1e-3)}")