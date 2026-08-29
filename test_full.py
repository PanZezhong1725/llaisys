import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import llaisys
from pathlib import Path
from transformers import AutoTokenizer

model_path = Path("D:/models/DeepSeek-R1-Distill-Qwen-1.5B")
print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Creating model...", flush=True)
model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
print("Model created successfully!", flush=True)

print("\nTesting inference...", flush=True)
prompt = "Who are you?"
input_content = tokenizer.apply_chat_template(
    conversation=[{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=False,
)
inputs = tokenizer.encode(input_content)
print(f"Input tokens: {inputs}", flush=True)

print("\nRunning generate...", flush=True)
outputs = model.generate(inputs, max_new_tokens=10)
print(f"Output tokens: {outputs}", flush=True)

print("\nDecoding output...", flush=True)
result = tokenizer.decode(outputs, skip_special_tokens=True)
print(f"Result: {result}", flush=True)

print("\nTest completed!", flush=True)
