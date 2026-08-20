from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

model_path = "/home/wnyxvo/huggingface/DeepSeek-R1-Distill-Qwen-1.5B"

def main():
	config = AutoConfig.from_pretrained(model_path)
	
	# 在 init_empty_weights 上下文中实例化模型
	with init_empty_weights():
		# 此时模型被创建在 "meta" 设备上，瞬间完成，内存占用为 0！
		model = AutoModelForCausalLM.from_config(config)

	# 打印整个模型
	print(model)

	# 对实现 LLAISYS 最有帮助：只看一个 Decoder Layer
	print("\n========== Decoder Layer 0 ==========")
	print(model.model.layers[0])

	print("\n========== Decoder Layer 0  打印每个权重名称和形状 ==========")
	for name, param in model.named_parameters():
		if name.startswith("model.layers.0."):
				print(f"{name:70s} {tuple(param.shape)}")

if __name__ == "__main__":
	main()