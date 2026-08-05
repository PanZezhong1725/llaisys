import copy
from pathlib import Path

import torch
from torchview import draw_graph
from transformers import AutoConfig, AutoModelForCausalLM


MODEL_PATH = "/home/wnyxvo/huggingface/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR = Path("./model_graph")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 1. 读取原始模型配置
original_config = AutoConfig.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

# 不修改原始配置对象
config = copy.deepcopy(original_config)


# 2. 为了让计算图可读，只保留一个 Decoder Layer
config.num_hidden_layers = 1

# 缩小 embedding 和 lm_head，避免它们占用大量内存
# Attention 和 MLP 的维度仍然保持真实模型配置
config.vocab_size = 256

config.use_cache = False

# Torchview 对 eager attention 通常兼容性更好
config._attn_implementation = "eager"

# 可视化使用 float32 CPU
config.dtype = torch.float32
if hasattr(config, "dtype"):
    config.dtype = torch.float32


# 3. 不要使用 init_empty_weights
# 创建一个真正位于 CPU 的缩小模型
model = AutoModelForCausalLM.from_config(config)
model = model.eval().cpu()


# 4. 构造一个很短的输入
batch_size = 1
sequence_length = 4

input_ids = torch.zeros(
    (batch_size, sequence_length),
    dtype=torch.long,
    device="cpu",
)

attention_mask = torch.ones_like(input_ids)


# 5. 生成计算图
graph = draw_graph(
    model,
    input_data={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": False,
    },
    device="cpu",
    expand_nested=True,
    depth=5,
    graph_name="deepseek_qwen2_single_layer",
)


# 6. 导出 SVG
output_path = OUTPUT_DIR / "deepseek_qwen2_single_layer"

graph.visual_graph.render(
    filename=str(output_path),
    format="svg",
    cleanup=True,
)

print(f"Model graph saved to: {output_path}.svg")