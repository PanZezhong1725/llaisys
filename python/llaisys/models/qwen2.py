from typing import Sequence
from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    llaisysQwen2Model_t,
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
)
from ..tensor import Tensor
from ctypes import c_int, c_int64, c_size_t, POINTER, byref

from pathlib import Path
import safetensors
import json


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # Load config
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Model hyperparameters
        self._dtype = DataType.BF16  # Default to BF16
        self._nlayer = config.get("num_hidden_layers", 28)
        self._hs = config.get("hidden_size", 1536)
        self._nh = config.get("num_attention_heads", 12)
        self._nkvh = config.get("num_key_value_heads", 2)
        self._dh = self._hs // self._nh
        self._di = config.get("intermediate_size", 8960)
        self._maxseq = config.get("max_position_embeddings", 32768)
        self._voc = config.get("vocab_size", 151936)
        self._epsilon = config.get("rms_norm_eps", 1e-6)
        self._theta = config.get("rope_theta", 10000.0)
        self._end_token = config.get("eos_token_id", 151643)
        
        self._device = device
        self._device_id = 0
        
        # Create C model
        meta = LlaisysQwen2Meta()
        meta.dtype = self._dtype
        meta.nlayer = self._nlayer
        meta.hs = self._hs
        meta.nh = self._nh
        meta.nkvh = self._nkvh
        meta.dh = self._dh
        meta.di = self._di
        meta.maxseq = self._maxseq
        meta.voc = self._voc
        meta.epsilon = self._epsilon
        meta.theta = self._theta
        meta.end_token = self._end_token
        
        device_ids = (c_int * 1)(self._device_id)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            device,
            device_ids,
            1
        )
        
        # Get weights pointer
        self._weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._weights = self._weights_ptr.contents
        
        # Load weights from safetensors
        self._load_weights(model_path)
    
    def _bf16_to_f32(self, bf16_array):
        """Convert bfloat16 numpy array to float32"""
        import numpy as np
        # bfloat16: 1 sign bit, 8 exponent bits, 7 mantissa bits
        # float32: 1 sign bit, 8 exponent bits, 23 mantissa bits
        # We need to shift the mantissa and preserve sign/exponent
        bf16_uint16 = bf16_array.view(np.uint16)
        
        # Process in chunks to avoid memory issues
        chunk_size = 10000
        result = np.empty(bf16_array.shape, dtype=np.float32)
        
        flat_bf16 = bf16_uint16.flat
        flat_result = result.flat
        
        for i in range(0, bf16_array.size, chunk_size):
            end = min(i + chunk_size, bf16_array.size)
            chunk = flat_bf16[i:end]
            
            # Extract sign, exponent, mantissa from bfloat16
            sign = (chunk >> 15).astype(np.uint32) & 0x1
            exponent = (chunk >> 7).astype(np.uint32) & 0xFF
            mantissa = chunk.astype(np.uint32) & 0x7F
            
            # Construct float32
            f32_chunk = (sign << 31) | (exponent << 23) | (mantissa << 16)
            flat_result[i:end] = f32_chunk.view(np.float32)
        
        return result
    
    def _load_weights(self, model_path: Path):
        """Load model weights from safetensors files"""
        import numpy as np
        import gc
        
        for file in sorted(model_path.glob("*.safetensors")):
            print(f"Loading {file.name}...", flush=True)
            
            # Read safetensors header manually to avoid loading entire file
            with open(file, "rb") as f:
                # Read header size (first 8 bytes)
                header_size = int.from_bytes(f.read(8), "little")
                # Read header JSON
                header_json = f.read(header_size).decode("utf-8")
                import json
                header = json.loads(header_json)
                
                # Get data start position
                data_start = 8 + header_size
                
                # Process each tensor
                keys = [k for k in header.keys() if k != "__metadata__"]
                print(f"  Found {len(keys)} tensors", flush=True)
                
                for i, name_ in enumerate(keys):
                    if i % 50 == 0:
                        print(f"  Loading tensor {i+1}/{len(keys)}: {name_}", flush=True)
                    
                    info = header[name_]
                    dtype_str = info["dtype"]
                    shape = info["shape"]
                    data_offsets = info["data_offsets"]
                    
                    # Read tensor data in small chunks to avoid memory issues
                    data_size = data_offsets[1] - data_offsets[0]
                    chunk_size = 4 * 1024 * 1024  # 4MB chunks
                    
                    # Determine numpy dtype
                    if dtype_str == "BF16":
                        np_dtype = np.uint16
                    elif dtype_str == "F32":
                        np_dtype = np.float32
                    elif dtype_str == "F16":
                        np_dtype = np.float16
                    elif dtype_str == "I64":
                        np_dtype = np.int64
                    else:
                        print(f"    Warning: Unsupported dtype {dtype_str}, skipping", flush=True)
                        continue
                    
                    # Calculate number of elements
                    numel = 1
                    for s in shape:
                        numel *= s
                    
                    # Read and process in chunks
                    if data_size > chunk_size:
                        # Large tensor - process in chunks
                        chunks = []
                        bytes_read = 0
                        while bytes_read < data_size:
                            f.seek(data_start + data_offsets[0] + bytes_read)
                            read_size = min(chunk_size, data_size - bytes_read)
                            chunk_bytes = f.read(read_size)
                            chunk_data = np.frombuffer(chunk_bytes, dtype=np_dtype)
                            chunks.append(chunk_data)
                            bytes_read += read_size
                            del chunk_bytes
                            del chunk_data
                        
                        # Concatenate chunks
                        tensor_data = np.concatenate(chunks).reshape(shape)
                        del chunks
                    else:
                        # Small tensor - read all at once
                        f.seek(data_start + data_offsets[0])
                        data_bytes = f.read(data_size)
                        tensor_data = np.frombuffer(data_bytes, dtype=np_dtype).reshape(shape)
                        del data_bytes
                    
                    self._load_weight(name_, tensor_data)
                    
                    # Free memory immediately
                    del tensor_data
                    gc.collect()
                    
                    if i % 50 == 0:
                        print(f"  Progress: {i+1}/{len(keys)} tensors loaded", flush=True)
            print(f"  Finished {file.name}", flush=True)
    
    def _bf16_to_f32(self, bf16_array):
        """Convert bfloat16 numpy array to float32"""
        import numpy as np
        # bfloat16: 1 sign bit, 8 exponent bits, 7 mantissa bits
        # float32: 1 sign bit, 8 exponent bits, 23 mantissa bits
        # We need to shift the mantissa and preserve sign/exponent
        bf16_uint16 = bf16_array.view(np.uint16)
        
        # Process in chunks to avoid memory issues
        chunk_size = 10000
        result = np.empty(bf16_array.shape, dtype=np.float32)
        
        flat_bf16 = bf16_uint16.flat
        flat_result = result.flat
        
        for i in range(0, bf16_array.size, chunk_size):
            end = min(i + chunk_size, bf16_array.size)
            chunk = flat_bf16[i:end]
            
            # Extract sign, exponent, mantissa from bfloat16
            sign = (chunk >> 15).astype(np.uint32) & 0x1
            exponent = (chunk >> 7).astype(np.uint32) & 0xFF
            mantissa = chunk.astype(np.uint32) & 0x7F
            
            # Construct float32
            f32_chunk = (sign << 31) | (exponent << 23) | (mantissa << 16)
            flat_result[i:end] = f32_chunk.view(np.float32)
        
        return result
    
    def _load_weight(self, name: str, data):
        """Load a single weight tensor"""
        # Map weight names to model structure
        if name == "model.embed_tokens.weight":
            self._copy_to_tensor(self._weights.in_embed, data)
        elif name == "lm_head.weight":
            self._copy_to_tensor(self._weights.out_embed, data)
        elif name == "model.norm.weight":
            self._copy_to_tensor(self._weights.out_norm_w, data)
        elif name.startswith("model.layers."):
            parts = name.split(".")
            layer_idx = int(parts[2])
            weight_name = ".".join(parts[3:])
            
            if weight_name == "input_layernorm.weight":
                self._copy_to_tensor(self._weights.attn_norm_w[layer_idx], data)
            elif weight_name == "self_attn.q_proj.weight":
                self._copy_to_tensor(self._weights.attn_q_w[layer_idx], data)
            elif weight_name == "self_attn.q_proj.bias":
                self._copy_to_tensor(self._weights.attn_q_b[layer_idx], data)
            elif weight_name == "self_attn.k_proj.weight":
                self._copy_to_tensor(self._weights.attn_k_w[layer_idx], data)
            elif weight_name == "self_attn.k_proj.bias":
                self._copy_to_tensor(self._weights.attn_k_b[layer_idx], data)
            elif weight_name == "self_attn.v_proj.weight":
                self._copy_to_tensor(self._weights.attn_v_w[layer_idx], data)
            elif weight_name == "self_attn.v_proj.bias":
                self._copy_to_tensor(self._weights.attn_v_b[layer_idx], data)
            elif weight_name == "self_attn.o_proj.weight":
                self._copy_to_tensor(self._weights.attn_o_w[layer_idx], data)
            elif weight_name == "post_attention_layernorm.weight":
                self._copy_to_tensor(self._weights.mlp_norm_w[layer_idx], data)
            elif weight_name == "mlp.gate_proj.weight":
                self._copy_to_tensor(self._weights.mlp_gate_w[layer_idx], data)
            elif weight_name == "mlp.up_proj.weight":
                self._copy_to_tensor(self._weights.mlp_up_w[layer_idx], data)
            elif weight_name == "mlp.down_proj.weight":
                self._copy_to_tensor(self._weights.mlp_down_w[layer_idx], data)
    
    def _copy_to_tensor(self, tensor_handle, data):
        """Copy numpy data to llaisys tensor"""
        import numpy as np
        from ..libllaisys import LIB_LLAISYS
        
        # Ensure data is contiguous and in the right format
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        
        # Use the C API to load weight
        LIB_LLAISYS.llaisysQwen2LoadWeight(tensor_handle, data.ctypes.data, data.nbytes)
    
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """Generate tokens using the model"""
        if max_new_tokens is None:
            max_new_tokens = 128
        
        # Convert inputs to list
        tokens = list(inputs)
        
        # For now, use argmax sampling (top_k=1)
        # TODO: Implement top-k and top-p sampling
        
        generated = []
        current_tokens = tokens.copy()
        
        for _ in range(max_new_tokens):
            # Convert to C array
            ntoken = len(current_tokens)
            token_array = (c_int64 * ntoken)(*current_tokens)
            
            # Run inference
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                token_array,
                c_size_t(ntoken)
            )
            
            generated.append(next_token)
            current_tokens = [next_token]
            
            # Check for end token
            if next_token == self._end_token:
                break
        
        return tokens + generated
    
    def __del__(self):
        if hasattr(self, "_model") and self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
