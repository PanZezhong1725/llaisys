#!/usr/bin/env python3
"""打印当前 Qwen2/DeepSeek 模型的结构和基本信息。

默认使用 meta device 仅构建模型结构，不读取或分配模型权重。
如果需要打印实际加载后的模型，可使用 ``--load-weights``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "DeepSeek-R1-Distill-Qwen-1.5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="打印 Hugging Face 模型结构")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"本地模型目录（默认：{DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="加载真实权重；默认只在 meta device 上构建并打印结构",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="加载真实权重时使用的设备，例如 cpu 或 cuda:0",
    )
    return parser.parse_args()


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/")


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    config_path = model_path / "config.json"

    if not config_path.is_file():
        raise SystemExit(f"找不到模型配置文件：{config_path}")

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)

    if args.load_weights:
        weight_files = list(model_path.glob("*.safetensors"))
        lfs_pointers = [path for path in weight_files if is_git_lfs_pointer(path)]
        if not weight_files:
            raise SystemExit(f"目录中没有 Safetensors 权重：{model_path}")
        if lfs_pointers:
            names = ", ".join(path.name for path in lfs_pointers)
            raise SystemExit(
                f"权重仍是 Git LFS 指针（{names}），请先执行 `git lfs pull`。"
            )

        dtype = getattr(config, "torch_dtype", None) or torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(args.device)
        mode = f"真实权重，设备={args.device}，dtype={dtype}"
    else:
        # meta device 只保留参数的形状和类型，不为数十亿参数分配实际内存。
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        mode = "仅模型结构（meta device，未加载权重）"

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"模型目录：{model_path}")
    print(f"模型类型：{config.model_type}")
    print(f"架构：{', '.join(config.architectures or [])}")
    print(f"打印模式：{mode}")
    print(f"参数量：{parameter_count:,}（约 {parameter_count / 1e9:.3f}B）")
    print("\n=== 模型结构 ===")
    print(model)


if __name__ == "__main__":
    main()
