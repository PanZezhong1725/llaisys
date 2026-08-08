# LLAISYS Assignment #4 Report

## Summary

This pull request completes Assignment #4: CUDA integration for LLAISYS.

Implemented:

- CUDA Runtime APIs for device management, streams, memory allocation, and memory copies.
- NVIDIA device resource management with a cuBLAS handle.
- CUDA implementations for all required operators: add, argmax, embedding, linear, rearrange, rms_norm, rope, self_attention, and swiglu.
- CUDA inference support for the Qwen2 model through the existing tensor and operator dispatch layers.
- Xmake build configuration for NVIDIA (`--nv-gpu=y`) and MetaX MACA (`--metax-gpu=y`).

## Supported Platforms and Status

| Platform | Status |
| --- | --- |
| CPU | Passed. Assignment #0-#3 tests pass in CI. |
| NVIDIA RTX 4090 | Passed. Runtime, all operator tests, and full model inference test pass. |
| MetaX C500 (MACA) | Passed. Build with Makefile.metax; runtime, operator, and model inference tests pass. |

## Reproduction Procedure

### NVIDIA RTX 4090

Test environment:

- GPU: NVIDIA GeForce RTX 4090 D
- Driver: 570.124.06
- CUDA Toolkit: 12.8
- Python: 3.12
- PyTorch: NVIDIA build with CUDA 12.8 support
- Xmake: 3.1.0

Build:

```bash
xmake f --nv-gpu=y -c
xmake -r
xmake install
```

Run tests:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/python

python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia
python test/test_infer.py --model [model_path] --test --device nvidia
```

### MetaX C500 (pending hardware validation)
### MetaX C500 (MACA)

Test environment:

- MACA: 3.5.3
- Device architecture: xcore1000
- Python: conda 3.10 with torch 2.8.0+metax3.5.3.9

Build with the standalone MetaX Makefile:

```bash
make -f Makefile.metax -j4
make -f Makefile.metax install
```

Run tests with the same `--device nvidia` commands because MetaX reuses the `LLAISYS_DEVICE_NVIDIA` device type through the MACA CUDA-compatible runtime:

```bash
export MACA_PATH=/opt/maca-3.5.3
export MACA_HOME=/opt/maca-3.5.3
export LD_LIBRARY_PATH=/opt/maca-3.5.3/lib64:/opt/maca-3.5.3/mxgpu_llvm/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/python

python test/test_runtime.py --device nvidia
python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia
python test/test_infer.py --test --device nvidia
```

## Results

### CI

Latest Actions run succeeded on both platforms:

- Windows latest: success
- Ubuntu latest: success
- Assignment #0, #1, #2, and #3 all passed.

### RTX 4090

- Runtime test: passed.
- Operator tests: all 8 operators passed for F32, F16, and BF16.
- Model inference: `test/test_infer.py --test --device nvidia` passed with 128 generated tokens, matching PyTorch output exactly.

### MetaX C500

- Runtime test: passed.
- Operator tests: all 8 operators passed for F32, F16, and BF16.
- Model inference: `test/test_infer.py --test --device nvidia` passed with 128 generated tokens, matching PyTorch output exactly.

## Notable Fixes

- Added `src/llaisys/cuda_link.cu` so the shared library is device-linked with nvcc; without this, loading `libllaisys.so` fails with an undefined `__cudaRegisterLinkedBinary` symbol.
- Changed F16/BF16 linear to use `cublasGemmEx` with `CUBLAS_COMPUTE_32F` and float alpha/beta for FP32 accumulation.
- Fixed the CUDA self-attention causal mask when `kvlen > qlen` by using `key_limit = q_idx + (kvlen - qlen)`.
- Fixed the CUDA argmax second-stage reduction to handle more than 256 block results via a grid-stride loop.
- Fixed `test/ops/self_attention.py` so the causal mask is created on the same device as the query tensor.
- Added `Makefile.metax`, a standalone MetaX build using mxcc with `--maca-link -fgpu-rdc` and `-lruntime_cu`.
