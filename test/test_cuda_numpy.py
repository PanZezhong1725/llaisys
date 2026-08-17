"""CUDA/CUDA-compatible backend validation without a PyTorch dependency."""

import argparse
import ctypes
import gc
import math

import ml_dtypes
import numpy as np

import llaisys


DTYPES = (
    ("f32", np.float32, llaisys.DataType.F32, 2e-5),
    ("f16", np.float16, llaisys.DataType.F16, 5e-3),
    ("bf16", ml_dtypes.bfloat16, llaisys.DataType.BF16, 3e-2),
)


class CudaHarness:
    def __init__(self):
        self.api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
        if self.api.get_device_count() < 1:
            raise RuntimeError("No CUDA-compatible device found.")
        self.api.set_device(0)

    def tensor_from(self, array, dtype):
        array = np.ascontiguousarray(array)
        tensor = llaisys.Tensor(
            array.shape, dtype, llaisys.DeviceType.NVIDIA, 0
        )
        tensor.load(ctypes.c_void_p(array.ctypes.data))
        return tensor

    @staticmethod
    def tensor_empty(shape, dtype):
        return llaisys.Tensor(shape, dtype, llaisys.DeviceType.NVIDIA, 0)

    def to_numpy(self, tensor, dtype):
        result = np.empty(tensor.shape(), dtype=dtype)
        self.api.set_device(tensor.device_id())
        self.api.memcpy_sync(
            ctypes.c_void_p(result.ctypes.data),
            tensor.data_ptr(),
            result.nbytes,
            llaisys.MemcpyKind.D2H,
        )
        return result


def assert_close(got, expected, tolerance):
    np.testing.assert_allclose(
        got.astype(np.float32),
        expected.astype(np.float32),
        atol=tolerance,
        rtol=0.0,
    )


def run_dtype_cases(harness, rng, name, np_dtype, dtype, tolerance, large):
    # Add.
    a = (rng.standard_normal((17, 19)) * 0.2).astype(np_dtype)
    b = (rng.standard_normal((17, 19)) * 0.2).astype(np_dtype)
    a_ = harness.tensor_from(a, dtype)
    b_ = harness.tensor_from(b, dtype)
    out_ = harness.tensor_empty(a.shape, dtype)
    llaisys.Ops.add(out_, a_, b_)
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np_dtype)
    assert_close(harness.to_numpy(out_, np_dtype), expected, tolerance)
    del out_, b_, a_

    # Argmax verifies both outputs, not merely one of them.
    values = (rng.standard_normal(4097) * 0.3).astype(np_dtype)
    values_ = harness.tensor_from(values, dtype)
    max_idx_ = harness.tensor_empty((1,), llaisys.DataType.I64)
    max_val_ = harness.tensor_empty((1,), dtype)
    llaisys.Ops.argmax(max_idx_, max_val_, values_)
    expected_idx = int(np.argmax(values.astype(np.float32)))
    assert harness.to_numpy(max_idx_, np.int64)[0] == expected_idx
    assert harness.to_numpy(max_val_, np_dtype)[0] == values[expected_idx]
    del max_val_, max_idx_, values_

    # Embedding.
    weight = (rng.standard_normal((23, 13)) * 0.1).astype(np_dtype)
    indices = np.array([0, 7, 22, 3, 3], dtype=np.int64)
    weight_ = harness.tensor_from(weight, dtype)
    indices_ = harness.tensor_from(indices, llaisys.DataType.I64)
    out_ = harness.tensor_empty((indices.size, weight.shape[1]), dtype)
    llaisys.Ops.embedding(out_, indices_, weight_)
    assert np.array_equal(harness.to_numpy(out_, np_dtype), weight[indices])
    del out_, indices_, weight_

    # Linear with and without bias.
    x = (rng.standard_normal((7, 11)) * 0.1).astype(np_dtype)
    weight = (rng.standard_normal((9, 11)) * 0.1).astype(np_dtype)
    bias = (rng.standard_normal(9) * 0.05).astype(np_dtype)
    x_ = harness.tensor_from(x, dtype)
    weight_ = harness.tensor_from(weight, dtype)
    bias_ = harness.tensor_from(bias, dtype)
    for use_bias in (True, False):
        out_ = harness.tensor_empty((7, 9), dtype)
        llaisys.Ops.linear(out_, x_, weight_, bias_ if use_bias else None)
        expected = x.astype(np.float32) @ weight.astype(np.float32).T
        if use_bias:
            expected += bias.astype(np.float32)
        assert_close(
            harness.to_numpy(out_, np_dtype),
            expected.astype(np_dtype),
            tolerance,
        )
        del out_
    del bias_, weight_, x_

    # K == 0 must return the broadcast bias, or zero when bias is absent.
    x = np.empty((2, 0), dtype=np_dtype)
    weight = np.empty((3, 0), dtype=np_dtype)
    bias = np.array([1, 2, 3], dtype=np_dtype)
    x_ = harness.tensor_from(x, dtype)
    weight_ = harness.tensor_from(weight, dtype)
    bias_ = harness.tensor_from(bias, dtype)
    for use_bias in (True, False):
        out_ = harness.tensor_empty((2, 3), dtype)
        llaisys.Ops.linear(out_, x_, weight_, bias_ if use_bias else None)
        expected = (
            np.broadcast_to(bias, (2, 3))
            if use_bias
            else np.zeros((2, 3), dtype=np_dtype)
        )
        assert np.array_equal(harness.to_numpy(out_, np_dtype), expected)
        del out_
    del bias_, weight_, x_

    # 65,536 rows force the capped RMSNorm grid to process a second row.
    rows = 65536 if large else 513
    x = (rng.standard_normal((rows, 4)) * 0.2).astype(np_dtype)
    weight = (rng.random(4) * 0.2).astype(np_dtype)
    x_ = harness.tensor_from(x, dtype)
    weight_ = harness.tensor_from(weight, dtype)
    out_ = harness.tensor_empty(x.shape, dtype)
    llaisys.Ops.rms_norm(out_, x_, weight_, 1e-5)
    x_f32 = x.astype(np.float32)
    expected = (
        x_f32
        / np.sqrt(np.mean(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-5)
        * weight.astype(np.float32)
    ).astype(np_dtype)
    assert_close(harness.to_numpy(out_, np_dtype), expected, tolerance)
    del out_, weight_, x_

    # RoPE.
    x = (rng.standard_normal((3, 2, 8)) * 0.2).astype(np_dtype)
    positions = np.array([0, 17, 1024], dtype=np.int64)
    x_ = harness.tensor_from(x, dtype)
    positions_ = harness.tensor_from(positions, llaisys.DataType.I64)
    out_ = harness.tensor_empty(x.shape, dtype)
    llaisys.Ops.rope(out_, x_, positions_, 10000.0)
    x_f32 = x.astype(np.float32)
    expected = np.empty_like(x_f32)
    for seq in range(3):
        for pair in range(4):
            angle = float(positions[seq]) / (10000.0 ** (2.0 * pair / 8.0))
            cosine, sine = math.cos(angle), math.sin(angle)
            expected[seq, :, pair] = (
                x_f32[seq, :, pair] * cosine
                - x_f32[seq, :, pair + 4] * sine
            )
            expected[seq, :, pair + 4] = (
                x_f32[seq, :, pair + 4] * cosine
                + x_f32[seq, :, pair] * sine
            )
    assert_close(
        harness.to_numpy(out_, np_dtype), expected.astype(np_dtype), tolerance
    )
    del out_, positions_, x_

    # GQA attention with value_dim crossing the 256-wide tile.
    query = (rng.standard_normal((2, 2, 7)) * 0.1).astype(np_dtype)
    key = (rng.standard_normal((3, 1, 7)) * 0.1).astype(np_dtype)
    value = (rng.standard_normal((3, 1, 257)) * 0.1).astype(np_dtype)
    query_ = harness.tensor_from(query, dtype)
    key_ = harness.tensor_from(key, dtype)
    value_ = harness.tensor_from(value, dtype)
    out_ = harness.tensor_empty((2, 2, 257), dtype)
    scale = 1.0 / math.sqrt(7.0)
    llaisys.Ops.self_attention(out_, query_, key_, value_, scale)
    expected = np.empty((2, 2, 257), dtype=np.float32)
    query_f32 = query.astype(np.float32)
    key_f32 = key.astype(np.float32)
    value_f32 = value.astype(np.float32)
    for query_idx in range(2):
        allowed_kv = 2 + query_idx
        for head in range(2):
            scores = key_f32[:allowed_kv, 0] @ query_f32[query_idx, head]
            scores *= scale
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            expected[query_idx, head] = probabilities @ value_f32[:allowed_kv, 0]
    assert_close(
        harness.to_numpy(out_, np_dtype), expected.astype(np_dtype), tolerance
    )
    del out_, value_, key_, query_

    # SwiGLU.
    gate = (rng.standard_normal((17, 19)) * 0.2).astype(np_dtype)
    up = (rng.standard_normal((17, 19)) * 0.2).astype(np_dtype)
    gate_ = harness.tensor_from(gate, dtype)
    up_ = harness.tensor_from(up, dtype)
    out_ = harness.tensor_empty(gate.shape, dtype)
    llaisys.Ops.swiglu(out_, gate_, up_)
    gate_f32 = gate.astype(np.float32)
    expected = (
        up.astype(np.float32) * gate_f32 / (1.0 + np.exp(-gate_f32))
    ).astype(np_dtype)
    assert_close(harness.to_numpy(out_, np_dtype), expected, tolerance)
    del out_, up_, gate_

    print(f"eight_ops_and_boundaries_{name}: PASS")


def run_long_attention(harness, rng):
    for name, np_dtype, dtype, tolerance in DTYPES:
        query = (rng.standard_normal((1, 1, 4)) * 0.1).astype(np_dtype)
        key = (rng.standard_normal((32769, 1, 4)) * 0.1).astype(np_dtype)
        value = (rng.standard_normal((32769, 1, 4)) * 0.1).astype(np_dtype)
        query_ = harness.tensor_from(query, dtype)
        key_ = harness.tensor_from(key, dtype)
        value_ = harness.tensor_from(value, dtype)
        out_ = harness.tensor_empty((1, 1, 4), dtype)
        llaisys.Ops.self_attention(out_, query_, key_, value_, 0.5)

        scores = key.astype(np.float32)[:, 0] @ query.astype(np.float32)[0, 0]
        scores *= 0.5
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        expected = (probabilities @ value.astype(np.float32)[:, 0]).astype(
            np_dtype
        )
        assert_close(
            harness.to_numpy(out_, np_dtype)[0, 0], expected, tolerance
        )
        del out_, value_, key_, query_
        print(f"long_attention_32769_{name}: PASS")


def run_linear_cancellation(harness):
    x = np.array([[300.0, 300.0]], dtype=np.float16)
    weight = np.array([[120.0, 120.0]], dtype=np.float16)
    bias = np.array([-64992.0], dtype=np.float16)
    x_ = harness.tensor_from(x, llaisys.DataType.F16)
    weight_ = harness.tensor_from(weight, llaisys.DataType.F16)
    bias_ = harness.tensor_from(bias, llaisys.DataType.F16)
    out_ = harness.tensor_empty((1, 1), llaisys.DataType.F16)
    llaisys.Ops.linear(out_, x_, weight_, bias_)
    result = harness.to_numpy(out_, np.float16)[0, 0]
    assert result == np.float16(7008.0), result
    print("linear_f16_cancellation_7008: PASS")


def run_large_add(harness):
    # One more element than a 65,535-block, 256-thread launch can cover.
    numel = 65535 * 256 + 1
    a = np.full((numel,), 1.25, dtype=np.float32)
    b = np.full((numel,), -0.5, dtype=np.float32)
    a_ = harness.tensor_from(a, llaisys.DataType.F32)
    b_ = harness.tensor_from(b, llaisys.DataType.F32)
    out_ = harness.tensor_empty(a.shape, llaisys.DataType.F32)
    llaisys.Ops.add(out_, a_, b_)
    assert np.all(harness.to_numpy(out_, np.float32) == np.float32(0.75))
    print(f"add_grid_stride_{numel}: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip memory-heavy grid-stride boundary cases.",
    )
    args = parser.parse_args()

    harness = CudaHarness()
    rng = np.random.default_rng(20260810)
    for dtype_spec in DTYPES:
        run_dtype_cases(harness, rng, *dtype_spec, large=not args.quick)
    run_long_attention(harness, rng)
    run_linear_cancellation(harness)
    if not args.quick:
        gc.collect()
        run_large_add(harness)
    print("CUDA_NUMPY_VALIDATION: PASS")


if __name__ == "__main__":
    main()
