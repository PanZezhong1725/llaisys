import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import llaisys
from pathlib import Path

def main():
    model_path = Path("D:/models/DeepSeek-R1-Distill-Qwen-1.5B")
    
    print("=" * 50)
    print("LLAISYS 简单推理测试")
    print("=" * 50)
    
    # 创建模型
    print("\n[1/2] 创建模型...", flush=True)
    model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
    print("[OK] 模型创建成功", flush=True)
    
    # 测试推理
    print("\n[2/2] 测试推理...", flush=True)
    inputs = [151646]  # 单个 token
    print(f"输入 tokens: {inputs}", flush=True)
    
    print("\n正在生成...", flush=True)
    outputs = model.generate(inputs, max_new_tokens=5)
    print(f"输出 tokens: {outputs}", flush=True)
    
    print("\n" + "=" * 50)
    print("推理完成!")
    print("=" * 50)
    print(f"输入: {inputs}")
    print(f"输出: {outputs}")
    print("=" * 50)

if __name__ == "__main__":
    main()
