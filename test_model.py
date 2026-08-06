import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import llaisys
from pathlib import Path

model_path = Path("D:/models/DeepSeek-R1-Distill-Qwen-1.5B")
print("Creating model...", flush=True)
try:
    model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
    print("Model created successfully!", flush=True)
except Exception as e:
    print(f"Error creating model: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Testing inference...", flush=True)
try:
    inputs = [151646, 151644, 15191, 525, 498, 30, 151645, 151648]
    print(f"Input tokens: {inputs}", flush=True)
    print("Running generate...", flush=True)
    outputs = model.generate(inputs, max_new_tokens=5)
    print(f"Output tokens: {outputs}", flush=True)
    print("Test completed!", flush=True)
except Exception as e:
    print(f"Error during inference: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
