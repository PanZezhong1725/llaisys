"""Diagnose CPU prefill: compare LLAISYS and HF logits after prefill."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))

import torch
import numpy as np
import ctypes
import llaisys
from transformers import AutoTokenizer, AutoModelForCausalLM
import safetensors.torch
from pathlib import Path

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
VOCAB_SIZE = 151936

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
prompt = "Who are you?"
input_content = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": prompt}],
    add_generation_prompt=True, tokenize=False)
inputs = tokenizer.encode(input_content)
print(f"Input tokens ({len(inputs)}): {inputs}")

# Load HF model (float32 for fair comparison)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True)

# HF forward — just prefill
with torch.no_grad():
    hf_input = torch.tensor([inputs], dtype=torch.long)
    hf_out = hf_model(hf_input, output_hidden_states=False)
    hf_logits = hf_out.logits[0, -1, :].numpy()

print(f"\nHF logits (prefill, last token): shape={hf_logits.shape}")
print(f"  max arg={np.argmax(hf_logits)}, val={np.max(hf_logits):.4f}")
top5 = np.argsort(hf_logits)[-5:][::-1]
for idx in top5:
    txt = tokenizer.decode([idx])
    print(f"  token {idx:6d} ({txt!r:20s}): {hf_logits[idx]:.4f}")

# Load LLAISYS model
print("\nLoading LLAISYS model...")
model_path = None
import glob
snaps = sorted(glob.glob(os.path.expanduser("~") + "/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/*"))
if snaps:
    model_path = snaps[-1]
    print(f"Model path: {model_path}")

if not model_path or not Path(model_path).is_dir():
    print("Model not found locally, downloading...")
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(MODEL_ID)
    print(f"Downloaded to: {model_path}")

# LLAISYS forward — CPU
print("\nRunning LLAISYS prefill on CPU...")
model = llaisys.models.Qwen2(str(model_path), llaisys.DeviceType.CPU)

# Create input tensor on CPU
input_ids_np = np.array(inputs, dtype=np.int64)
input_tensor = llaisys.Tensor([len(inputs)], llaisys.DataType.I64, llaisys.DeviceType.CPU, 0)
api = llaisys.RuntimeAPI(llaisys.DeviceType.CPU)
api.memcpy_sync(input_tensor.data_ptr(), input_ids_np.ctypes.data, len(inputs) * 8, llaisys.MemcpyKind.H2D)

logits_tensor = llaisys.Tensor([VOCAB_SIZE], llaisys.DataType.F32, llaisys.DeviceType.CPU, 0)
llaisys.LIB_LLAISYS.qwen2Forward(model._model, input_tensor._tensor, logits_tensor._tensor)

logits_np = np.ctypeslib.as_array(
    ctypes.cast(logits_tensor.data_ptr(), ctypes.POINTER(ctypes.c_float)),
    shape=(VOCAB_SIZE,)
).copy()

print(f"\nLLAISYS logits (prefill, last token): shape={logits_np.shape}")
print(f"  max arg={np.argmax(logits_np)}, val={np.max(logits_np):.4f}")
top5 = np.argsort(logits_np)[-5:][::-1]
for idx in top5:
    txt = tokenizer.decode([idx])
    print(f"  token {idx:6d} ({txt!r:20s}): {logits_np[idx]:.4f}")

# Compare
print("\n--- Comparison ---")
diff = logits_np - hf_logits
print(f"max abs diff: {np.max(np.abs(diff)):.6f}")
print(f"HF argmax: {np.argmax(hf_logits)}, LLAISYS argmax: {np.argmax(logits_np)}")
print(f"Match: {np.argmax(hf_logits) == np.argmax(logits_np)}")

# Check if first transformer layer's attention output matches
# (would need deeper instrumentation to compare intermediate tensors)