"""Test script to verify Qwen2 model loading and basic inference via LLAISYS."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from llaisys.models.qwen2 import Qwen2
from llaisys.libllaisys import DeviceType
from transformers import AutoTokenizer
import time


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "D:\\models\\DeepSeek-R1-Distill-Qwen-1.5B"
    device_name = sys.argv[2] if len(sys.argv) > 2 else "cpu"

    device = (
        DeviceType.CPU
        if device_name == "cpu"
        else DeviceType.NVIDIA
        if device_name == "nvidia"
        else DeviceType.SUDA
    )

    print(f"Loading model from: {model_path}")
    print(f"Device: {device_name}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load LLAISYS model
    print("Loading LLAISYS Qwen2 model...")
    start = time.time()
    model = Qwen2(model_path, device=device)
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    # Test with a simple prompt
    prompt = "Hello, who are you?"
    print(f"\nPrompt: {prompt}")

    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)

    print(f"Input tokens ({len(inputs)}): {inputs[:10]}...")

    # Generate
    print("\nGenerating...")
    start = time.time()
    outputs = model.generate(
        inputs,
        max_new_tokens=32,
        top_k=50,
        top_p=0.8,
        temperature=0.8,
    )
    gen_time = time.time() - start

    result = tokenizer.decode(outputs, skip_special_tokens=True)
    print(f"\n=== Result ===")
    print(f"Output tokens ({len(outputs)}): {outputs}")
    print(f"Generated text: {result}")
    print(f"\nGeneration time: {gen_time:.2f}s")
    print(f"Tokens/sec: {len(outputs) / gen_time:.2f}")

    print("\n\033[92mTest completed successfully!\033[0m")


if __name__ == "__main__":
    main()
