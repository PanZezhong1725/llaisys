import json
import struct
from pathlib import Path


def read_safetensors_header(path: Path) -> dict:
	"""读取并解析单张量文件的 JSON 头部元数据"""
	with path.open("rb") as file:
		header_size_bytes = file.read(8)
		if len(header_size_bytes) != 8:
			raise ValueError(f"Invalid safetensors file: {path}")

		header_size = struct.unpack("<Q", header_size_bytes)[0]
		header = file.read(header_size)

	return json.loads(header)


def main():
	""" 不加载数据，直接读取 safetensors 文件头 """
	model_dir = Path("/home/wnyxvo/huggingface/DeepSeek-R1-Distill-Qwen-1.5B")

	if not model_dir.exists():
		print(f"❌ 错误: 找不到目录 {model_dir}")
		return

	# 遍历并打印所有分片信息
	for shard in sorted(model_dir.glob("*.safetensors")):
		print(f"\n========== {shard.name} ==========")

		try:
			header = read_safetensors_header(shard)
			for name, metadata in header.items():
				if name == "__metadata__":
					continue
				
				if name.startswith("model.layers.0."):
					print(
						f"{name:70s} "
						f"dtype={metadata['dtype']:5s} "
						f"shape={metadata['shape']}"
					)
		except Exception as e:
			print(f"读取文件 {shard.name} 失败: {e}")


if __name__ == "__main__":
	main()

