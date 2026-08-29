import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import llaisys
from pathlib import Path
from transformers import AutoTokenizer

def main():
    model_path = Path("D:/models/DeepSeek-R1-Distill-Qwen-1.5B")
    
    print("=" * 50)
    print("LLAISYS 模型推理测试")
    print("=" * 50)
    
    # 加载 tokenizer
    print("\n[1/3] 加载 tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("[OK] Tokenizer 加载成功", flush=True)
    
    # 创建模型
    print("\n[2/3] 创建模型...", flush=True)
    model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
    print("[OK] 模型创建成功", flush=True)
    
    # 测试推理
    print("\n[3/3] 测试推理...", flush=True)
    prompt = "Who are you?"
    print(f"输入提示: {prompt}", flush=True)
    
    # 编码输入
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    print(f"输入 tokens: {inputs}", flush=True)
    
    # 运行推理
    print("\n正在生成回复...", flush=True)
    outputs = model.generate(inputs, max_new_tokens=20)
    print(f"输出 tokens: {outputs}", flush=True)
    
    # 解码输出
    result = tokenizer.decode(outputs, skip_special_tokens=True)
    print("\n" + "=" * 50)
    print("推理结果:")
    print("=" * 50)
    print(result)
    print("=" * 50)
    
    print("\n[OK] 测试完成!", flush=True)

if __name__ == "__main__":
    main()
