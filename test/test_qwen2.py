import tempfile

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import llaisys


def reference_generate(model, inputs, max_new_tokens, eos_token_id):
    tokens = list(inputs)
    current = torch.tensor([tokens], dtype=torch.int64)
    cache = None
    with torch.no_grad():
        for _ in range(max_new_tokens):
            output = model(current, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            next_token = int(output.logits[0, -1].argmax())
            tokens.append(next_token)
            if next_token == eos_token_id:
                break
            current = torch.tensor([[next_token]], dtype=torch.int64)
    return tokens


def test_qwen2(dtype, tie_word_embeddings=False):
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=tie_word_embeddings,
    )
    model = Qwen2ForCausalLM(config).eval().to(dtype)
    inputs = [1, 5, 7, 9]
    max_new_tokens = 5

    with tempfile.TemporaryDirectory(prefix="llaisys-qwen2-") as model_path:
        model.save_pretrained(model_path, safe_serialization=True)
        expected = reference_generate(
            model, inputs, max_new_tokens, config.eos_token_id
        )
        llaisys_model = llaisys.models.Qwen2(model_path)
        actual = llaisys_model.generate(inputs, max_new_tokens=max_new_tokens)
        assert actual == expected, f"LLAISYS: {actual}\nPyTorch: {expected}"
        repeated = llaisys_model.generate(inputs, max_new_tokens=max_new_tokens)
        assert repeated == expected, "Resetting the KV cache changed generation"


if __name__ == "__main__":
    print("Testing Qwen2 inference on CPU")
    cases = (
        (torch.float32, False),
        (torch.bfloat16, False),
        (torch.float32, True),
    )
    for dtype, tied in cases:
        print(f"   dtype <{dtype}> tied embeddings <{tied}>")
        test_qwen2(dtype, tied)
    print("\033[92mTest passed!\033[0m\n")
