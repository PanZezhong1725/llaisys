"""Small, deterministic end-to-end test for the Qwen2 backend.

Unlike ``test_infer.py``, this test does not download the 1.5B checkpoint.  It
builds a tiny random Qwen2 checkpoint, compares greedy generation with an
independent NumPy implementation, and checks that chunked prefill produces the
same next token as one-shot prefill.  Run it as a script after installing the
local package.
"""

from ctypes import c_int64
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from safetensors.numpy import save_file

import llaisys
from llaisys.libllaisys import LIB_LLAISYS


CONFIG = {
    "torch_dtype": "float32",
    "num_hidden_layers": 2,
    "hidden_size": 8,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "intermediate_size": 12,
    "max_position_embeddings": 32,
    "vocab_size": 17,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "eos_token_id": 16,
}


def make_checkpoint(path: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260811)
    hs = CONFIG["hidden_size"]
    nh = CONFIG["num_attention_heads"]
    nkvh = CONFIG["num_key_value_heads"]
    dh = hs // nh
    di = CONFIG["intermediate_size"]
    voc = CONFIG["vocab_size"]

    def normal(shape, scale=0.2):
        return np.ascontiguousarray(
            rng.normal(0.0, scale, size=shape).astype(np.float32)
        )

    weights = {
        "model.embed_tokens.weight": normal((voc, hs)),
        "model.norm.weight": np.ascontiguousarray(
            (1.0 + rng.normal(0.0, 0.05, size=hs)).astype(np.float32)
        ),
        "lm_head.weight": normal((voc, hs)),
    }

    for layer in range(CONFIG["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        weights.update(
            {
                prefix + "input_layernorm.weight": np.ascontiguousarray(
                    (1.0 + rng.normal(0.0, 0.05, size=hs)).astype(np.float32)
                ),
                prefix + "self_attn.q_proj.weight": normal((nh * dh, hs)),
                prefix + "self_attn.q_proj.bias": normal((nh * dh,), 0.05),
                prefix + "self_attn.k_proj.weight": normal((nkvh * dh, hs)),
                prefix + "self_attn.k_proj.bias": normal((nkvh * dh,), 0.05),
                prefix + "self_attn.v_proj.weight": normal((nkvh * dh, hs)),
                prefix + "self_attn.v_proj.bias": normal((nkvh * dh,), 0.05),
                prefix + "self_attn.o_proj.weight": normal((hs, nh * dh)),
                prefix + "post_attention_layernorm.weight": np.ascontiguousarray(
                    (1.0 + rng.normal(0.0, 0.05, size=hs)).astype(np.float32)
                ),
                prefix + "mlp.gate_proj.weight": normal((di, hs)),
                prefix + "mlp.up_proj.weight": normal((di, hs)),
                prefix + "mlp.down_proj.weight": normal((hs, di)),
            }
        )

    (path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    save_file(weights, path / "model.safetensors")
    return weights


class NumpyQwen2:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.weights = weights
        self.hs = CONFIG["hidden_size"]
        self.nh = CONFIG["num_attention_heads"]
        self.nkvh = CONFIG["num_key_value_heads"]
        self.dh = self.hs // self.nh

    @staticmethod
    def linear(x, weight, bias=None):
        out = x @ weight.T
        return out if bias is None else out + bias

    @staticmethod
    def rms_norm(x, weight):
        variance = np.mean(x * x, axis=-1, keepdims=True)
        return x * (1.0 / np.sqrt(variance + CONFIG["rms_norm_eps"])) * weight

    def rope(self, x):
        half = self.dh // 2
        positions = np.arange(x.shape[0], dtype=np.float32)[:, None]
        dimensions = np.arange(half, dtype=np.float32)[None, :]
        angles = positions / (CONFIG["rope_theta"] ** (2.0 * dimensions / self.dh))
        cos = np.cos(angles)[:, None, :]
        sin = np.sin(angles)[:, None, :]
        first, second = x[..., :half], x[..., half:]
        return np.concatenate(
            (first * cos - second * sin, second * cos + first * sin),
            axis=-1,
        )

    def attention(self, hidden, layer):
        prefix = f"model.layers.{layer}."
        norm = self.rms_norm(
            hidden,
            self.weights[prefix + "input_layernorm.weight"],
        )
        q = self.linear(
            norm,
            self.weights[prefix + "self_attn.q_proj.weight"],
            self.weights[prefix + "self_attn.q_proj.bias"],
        ).reshape(-1, self.nh, self.dh)
        k = self.linear(
            norm,
            self.weights[prefix + "self_attn.k_proj.weight"],
            self.weights[prefix + "self_attn.k_proj.bias"],
        ).reshape(-1, self.nkvh, self.dh)
        v = self.linear(
            norm,
            self.weights[prefix + "self_attn.v_proj.weight"],
            self.weights[prefix + "self_attn.v_proj.bias"],
        ).reshape(-1, self.nkvh, self.dh)

        q = self.rope(q).transpose(1, 0, 2)
        k = np.repeat(self.rope(k), self.nh // self.nkvh, axis=1).transpose(1, 0, 2)
        v = np.repeat(v, self.nh // self.nkvh, axis=1).transpose(1, 0, 2)
        scores = np.einsum("htd,hsd->hts", q, k) / np.sqrt(np.float32(self.dh))
        scores = np.where(
            np.tril(np.ones(scores.shape[-2:], dtype=bool))[None, ...],
            scores,
            -np.inf,
        )
        scores -= np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        values = np.einsum("hts,hsd->htd", probabilities, v)
        values = values.transpose(1, 0, 2).reshape(-1, self.hs)
        projected = self.linear(
            values,
            self.weights[prefix + "self_attn.o_proj.weight"],
        )
        return hidden + projected

    def layer(self, hidden, layer):
        prefix = f"model.layers.{layer}."
        hidden = self.attention(hidden, layer)
        norm = self.rms_norm(
            hidden,
            self.weights[prefix + "post_attention_layernorm.weight"],
        )
        gate = self.linear(norm, self.weights[prefix + "mlp.gate_proj.weight"])
        up = self.linear(norm, self.weights[prefix + "mlp.up_proj.weight"])
        activated = (gate / (1.0 + np.exp(-gate))) * up
        down = self.linear(
            activated,
            self.weights[prefix + "mlp.down_proj.weight"],
        )
        return hidden + down

    def next_token(self, token_ids):
        hidden = self.weights["model.embed_tokens.weight"][token_ids]
        for layer in range(CONFIG["num_hidden_layers"]):
            hidden = self.layer(hidden, layer)
        hidden = self.rms_norm(hidden[-1:], self.weights["model.norm.weight"])
        logits = self.linear(hidden, self.weights["lm_head.weight"])
        return int(np.argmax(logits[0]))

    def generate(self, token_ids, max_new_tokens):
        result = list(token_ids)
        for _ in range(max_new_tokens):
            result.append(self.next_token(result))
            if result[-1] == CONFIG["eos_token_id"]:
                break
        return result


def backend_next_token_by_chunks(model, chunks):
    LIB_LLAISYS.llaisysQwen2ModelReset(model._model)
    result = None
    for chunk in chunks:
        token_buffer = (c_int64 * len(chunk))(*chunk)
        result = LIB_LLAISYS.llaisysQwen2ModelInfer(
            model._model,
            token_buffer,
            len(chunk),
        )
        assert result >= 0
    return result


def run_tiny_qwen2(device: llaisys.DeviceType):
    with tempfile.TemporaryDirectory(prefix="llaisys-qwen2-") as directory:
        path = Path(directory)
        weights = make_checkpoint(path)
        reference = NumpyQwen2(weights)
        model = llaisys.models.Qwen2(path, device)

        prompt = [1, 5, 3, 9]
        expected = reference.generate(prompt, max_new_tokens=6)
        actual = model.generate(prompt, max_new_tokens=6)
        assert actual == expected, (actual, expected)

        one_shot = backend_next_token_by_chunks(model, [prompt])
        token_by_token = backend_next_token_by_chunks(model, [[token] for token in prompt])
        split_prefill = backend_next_token_by_chunks(model, [prompt[:2], prompt[2:]])
        assert one_shot == token_by_token == split_prefill == reference.next_token(prompt)

        # A second public generation verifies that generate() resets stale cache.
        second_prompt = [2, 4, 7]
        assert model.generate(second_prompt, max_new_tokens=4) == reference.generate(
            second_prompt,
            max_new_tokens=4,
        )


def test_tiny_qwen2():
    run_tiny_qwen2(llaisys.DeviceType.CPU)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "nvidia"),
    )
    args = parser.parse_args()
    device = {
        "cpu": llaisys.DeviceType.CPU,
        "nvidia": llaisys.DeviceType.NVIDIA,
    }[args.device]
    run_tiny_qwen2(device)
    print("\033[92mTiny Qwen2 test passed!\033[0m")
