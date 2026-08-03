# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

LLAISYS ("Let's Learn AI SYStem") — an educational C++/Python AI systems project. C++ backend (`src/`) exposes a C API (`include/llaisys/*.h`), wrapped by Python ctypes (`python/llaisys/libllaisys/`) and a friendlier Python layer (`python/llaisys/`). Full assignment structure: `README.md` / `README_ZN.md`.

Build: `xmake` (compile) → `xmake install` (copies `.so` into `python/llaisys/libllaisys/`) → `pip install -e ./python` (editable install picks up `.py` changes immediately, no reinstall needed).

## Assignment #3 — Qwen2 inference: DONE

Goal: implement full Qwen2 (prefill + incremental decode) inference in C++, driven by a thin Python wrapper. No PyTorch allowed in the Python compute path — only for loading/comparing weights and running the HF reference in `test/test_infer.py`. Target model: DeepSeek-R1-Distill-Qwen-1.5B (28 layers, hs=1536, nh=12, nkvh=2, dh=128, di=8960, theta=10000, eps=1e-6, tie_word_embeddings=false). Full task breakdown: `docs/QWEN2_INFERENCE_ZH.md`.

`test/test_infer.py --model DeepSeek-R1-Distill-Qwen-1.5B --test` passes (`Test passed!`, greedy-decode token-for-token match against HF `transformers`). Two bugs fixed after the initial "implementation done" pass, both by the user with step-by-step guidance, not written for them:
- `python/llaisys/models/qwen2.py`: absolute import `from python.llaisys.libllaisys.llaisys_types import DataType` only worked when run as a script from repo root, not as an installed package — changed to relative import `from ..libllaisys.llaisys_types import DataType`.
- `python/llaisys/models/qwen2.py` `__init__`: `end_token` was hardcoded to `151646`, which is actually `bos_token_id` — the real `eos_token_id` (151643) was already sitting in the same `config` dict as the other meta fields, just needed `config["eos_token_id"]` instead of the literal.
- `python/llaisys/models/qwen2.py` `generate`: was returning only newly-generated tokens; the test expects the HF-`generate()` convention (prompt + generated tokens concatenated) — fixed by appending `next_token` to `generated_tokens` (the list already seeded with `list(inputs)`) each loop iteration instead of a separate now-dead `output_tokens` list, and returning `generated_tokens`.

## Next: Assignment #4 — CUDA integration

**Core correctness done.** Full task breakdown: `docs/CUDA_INTEGRATION_ZH.md`. This machine has a real GPU reachable from WSL2 (`nvidia-smi` shows an RTX 5070 Ti Laptop GPU, `nvcc` is CUDA 12.9) so this can be developed and tested locally, not just on camp-provided remote resources. `xmake build llaisys --nv-gpu=y` now succeeds end-to-end, and all 8 non-deprecated ops pass `python test/ops/<op>.py --device nvidia` (`add`, `embedding`, `argmax`, `rope`, `linear`, `swiglu`, `rms_norm`, `self_attention`). Remaining work is the performance-optimization pass below, not correctness.

### Per-op status (`src/ops/*/nvidia/*.cu`) — all done, wired, and passing `test/ops/*.py --device nvidia`

- **`add`, `embedding`**: done (elementwise, no reduction needed).
- **`argmax`**: done. Two-round design: round 1 launches with `grid_size = ceil(numel/block_size)` blocks, each block does a thread-local grid-stride scan + shared-memory tree reduction (writes one (val, idx) candidate per block into a `cudaMalloc`'d intermediate buffer); round 2 launches `<<<1, block_size>>>` over that buffer (`numel` becomes `grid_size`) to produce the final `max_idx`/`max_val`. `argmax_kernel` takes an extra `const int64_t *idx_map` param (`nullptr` in round 1, the round-1 index buffer in round 2) so round 2 can translate "position within the candidate buffer" back to the original vocab index — round 2's kernel invocation would otherwise report the wrong `max_idx` (right `max_val`, wrong index).
- **`rms_norm`**: thread-local sum-of-squares over the row → shared-memory tree reduction, accumulated/stored in `float` regardless of `T` for precision → `rms = sqrtf(shared[0]/d + eps)` read from `shared[0]` by every thread → normalize+scale+write loop using the same `row*d+i` indexing as the read phase.
- **`rearrange`**: deprecated, will not be implemented — `src/ops/rearrange/nvidia/rearrange_cuda.cu` stays a 0-line stub on purpose. Nothing in the NVIDIA path needs it: the KV-cache copy (the one place CPU code used it as a substitute) uses a flat `std::memcpy` instead, since both sides are provably contiguous. Don't propose implementing this op unless the user explicitly reopens it.
- **`rope`, `linear`, `swiglu`**: done. `linear`'s NVIDIA kernel is now the tiled-GEMM rewrite (see optimization plan below) — no longer the naive one-thread-per-output-element version.
- **`self_attention`**: V1 kernel in `src/ops/self_attention/nvidia/self_attention_cuda.cu` — one block per (query token `i`, head `h`) pair, causal + GQA-aware (grid-stride dot product + shared-memory tree reduction for `max_score`, in-place `exp`/sum-reduction for `sum_exp`, final weighted-sum-by-`dv` pass). Correct but known to be slow (see optimization plan below): materializes the full `scores[j]` row in shared memory and does two full-block reductions per (i,h) pair.

### Recurring bug class: `op.cpp` missing the `nvidia/*.cuh` include

Every op's `op.cpp` NVIDIA wiring hit the same two mistakes at least once — worth recognizing on sight rather than re-diagnosing each time:
1. **Missing include entirely**: `op.cpp` calls `cuda::foo(...)` but never `#include`s the header declaring `llaisys::ops::cuda::foo` — compiles to `error: 'cuda' has not been declared`. Fix: add `#include "nvidia/foo_cuda.cuh"` (op.cpp lives one directory *above* `nvidia/`, so the include needs that prefix — see `add`/`swiglu`'s `op.cpp` for the pattern that was always correct).
2. **Wrong prefix inside the `.cu` file itself**: the same `nvidia/foo_cuda.cuh` prefix is wrong *inside* `foo_cuda.cu`, since that file already lives in the `nvidia/` directory — it needs the bare `#include "foo_cuda.cuh"`. This bit `self_attention_cuda.cu` once when the `nvidia/` prefix was copy-pasted into the wrong file during the `op.cpp` fix.

This hit `linear`, `self_attention`, `rms_norm`, `argmax`, and `rope`'s `op.cpp` in turn — all now fixed.

### Test-script bug fixed along the way (not project code)

`test/ops/self_attention.py`'s `torch_self_attention` reference (line 18) built its causal-mask tensor with `torch.ones(L, S, dtype=torch.bool)` — no `device=`, so it defaulted to CPU while `attn_bias` (line 16) correctly used `device=query.device`. Under `--device cpu` this never surfaced; under `--device nvidia` it crashed with a device-mismatch `RuntimeError` in `masked_fill_`, unrelated to `self_attention_cuda.cu`'s correctness (already independently verified — see below). Fixed by adding `device=query.device` to the `torch.ones(...)` call.

### Performance optimization plan (linear + self_attention → flash attention)

- **`linear`: DONE.** `src/ops/linear/nvidia/linear_cuda.cu` rewritten to standard tiled-GEMM — static shared-memory `ins`/`weights` tiles (`__shared__ T ins[TILE_SIZE][TILE_SIZE]`, `TILE_SIZE=16` at file scope so both the kernel and launcher see it), one `for(t=0;t<numTiles;t++)` loop over `K`-chunks: cooperative load (each thread loads one element per tile, zero-padding on out-of-range `row`/`localcol`/`col`/`localrow`) → `__syncthreads()` → per-thread accumulation `sum += ins[threadIdx.y][k] * weights[k][threadIdx.x]` over the tile → second `__syncthreads()` before the next `t` overwrites the tiles → final bounds-checked write + bias. Verified against a from-scratch CPU reference via the standalone-nvcc-harness pattern (f32 + bf16, shapes spanning exact/non multiples of `TILE_SIZE` including `M`/`N`/`K` all non-16-multiples) and via `test/ops/linear.py --device nvidia`; both pass.
  - **Benchmarked vs. the old naive kernel** (scratchpad `bench.cu`, `cudaEvent` timing, naive kernel preserved in scratchpad `naive_linear_cuda.cu` under a renamed `cuda_naive::` namespace since the real one was overwritten in place): tiled is **2.7x-3.8x faster** for prefill-sized (`M=17`) and large/square shapes, but **~0.72x (28% slower) for `M=1` decode-step shapes** (e.g. the per-token QKV/out_embed projections during autoregressive generation). Root cause: `block_size` is `16x16`, so at `M=1` only 1 of each block's 16 thread-rows does useful work — the other 15 pay the full `__syncthreads()`/shared-memory-load overhead for zero benefit, since there's no cross-row reuse to amortize it against (classic tall-skinny-GEMM tiling trade-off). Since decode dominates total token count in real generation, this is worth revisiting eventually (e.g. dispatching to the naive kernel when `M` is small) but is out of scope for this pass — noted here, not fixed.
- **`self_attention`**, staged, roughly in order (not yet started):
  1. Replace the shared-memory tree reductions (max, then sum) with warp-level `__shfl_down_sync` reduction, falling back to shared memory only across warps (fewer full-block syncs).
  2. Change grid granularity from one block per single query row to one block per *tile* of query rows (e.g. 64/128) sharing head `h`, so a loaded K/V tile is reused across multiple query rows instead of re-read per row.
  3. **End goal: flash attention** — replace the current "materialize full `scores[]` row, then 3 full passes + 2 full-block reductions" structure with a single fused streaming pass over K/V tiles using online/running softmax (`running_max`, `running_sum`, `running_acc`, rescaled by `exp(old_max - new_max)` whenever a new tile raises the max), eliminating the O(`total_len`) shared-memory row entirely. This is a structural rewrite of the kernel body, not an incremental patch on steps 1-2 — causal-mask (`limit`/`causal_offset`) and GQA (`group`/`kvh`) bookkeeping carry over unchanged.

Same hands-off/step-by-step guidance rule applies to this work as the rest of Assignment #4 — see "Working style for this project" below.

### Testing approach

The full project now builds (`xmake build llaisys --nv-gpu=y`) and `python test/ops/<op>.py --device nvidia` is the primary end-to-end check — no more workaround needed for finished ops. The standalone-nvcc-harness pattern used earlier (compile one op's `.cu` file plus a small scratchpad `main()` directly via `nvcc`, comparing against a from-scratch CPU reference, bypassing the full build) is still useful for iterating on the optimization work above without a full rebuild each time — see `self_attention`'s and `rms_norm`'s scratchpad `main.cu` for the pattern (random-input cases spanning edge shapes like `d`/`total_len`/`dv` not a multiple of `block_size`, f32 + bf16, real-model shapes).

### What's implemented — status as of end of the Infer/Python-glue pass

All three pieces below are now implemented, compiled, and verified working end-to-end (manual `generate()` smoke test produced coherent text for a real prompt against the real DeepSeek-R1-Distill-Qwen-1.5B weights). Same hands-off/step-by-step-guidance rule still applies to any further changes in these files — don't rewrite user's logic for them, review/point-at-line-numbers instead.

- `src/llaisys/models/qwen2.cc`: `LlaisysQwen2Model` struct + `Create`/`Destroy`/`Weights`/`Infer` all implemented by the user (embedding → per-layer loop [rms_norm, qkv proj (2D out, matches `linear`'s shape assert), view to 3D, rope on q/k only, kv-cache write via `slice(cur_len,total_len)->load(...)`, kv-cache read via `slice(0,total_len)`, self_attention, attn_o proj, residual, mlp rms_norm, gate/up proj, swiglu, down proj, residual] → final rms_norm → out_embed proj → slice last token → argmax → `cur_len = total_len` → return). Two bugs found via self-review and fixed by the user: (1) `q`/`k`/`v` tensors were created 3D (`{ntoken, nh, dh}`) and passed straight into `ops::linear`, which asserts a 2D `out` shape matching `weight`'s dout — fixed by creating them 2D (`{ntoken, nh*dh}`) and keeping the existing `->view(...)` to reshape to 3D before RoPE; (2) `argmax`'s `max_val` tensor was created with `LLAISYS_DTYPE_I64` instead of `meta.dtype` — `argmax` asserts `max_val`'s dtype matches the input values' dtype, only `max_idx` must be I64.
- `python/llaisys/libllaisys/model.py` (singular, not `models.py`): ctypes bindings for `LlaisysQwen2Meta`/`LlaisysQwen2Weights`/the four `llaisysQwen2Model*` functions, wired into `__init__.py` via `load_model(lib)`. Done.
- `python/llaisys/models/qwen2.py`: both TODOs done.
  - `__init__`: builds `self.meta` (stored on `self`, needed by `generate()` for `end_token`), calls `Create`/`Weights`, then loads safetensors weights into the pre-allocated tensors via two lookup tables — `GLOBAL_MAP` (full key name → attribute on `weights.contents`, for `in_embed`/`out_embed`/`out_norm_w`) and `LAYER_MAP` (key with `"model.layers.{i}."` prefix stripped → attribute, indexed by the parsed-out layer number), then `LIB_LLAISYS.tensorLoad(handle, ptr)` per weight. Requires `import ml_dtypes` before `safetensors.safe_open(..., framework="numpy", ...)` — numpy has no native bf16 dtype, and `ml_dtypes` registers the extension dtype safetensors needs on import (installed via `pip install ml_dtypes`; without the import, even with the package installed, `get_tensor` raises `TypeError: data type 'bfloat16' not understood`).
  - `generate`: prefill and decode share one loop — `current_input` is the full prompt on the first iteration only, then always `[next_token]` after; `generated_tokens` (seeded with `list(inputs)`) has `next_token` appended each iteration and is what gets returned (matches HF `generate()`'s convention of prompt+generated concatenated); breaks on `next_token == self.meta.end_token`.
- Reference (non-compiled, not imported by anything) implementations the user can diff against once stuck: `docs/qwen2_infer_reference.cc`, `docs/qwen2_python_bindings_reference.py`, `docs/qwen2_model_reference.py`.

### Still open
- dtype/contiguity sanity-check TODO left as a comment in `qwen2.py`'s weight-loading loop was never actually implemented (low priority — real run already proves the bf16 raw-memcpy path works for this checkpoint).

### Op signatures already implemented (Assignment #2, reusable as-is)

All under `src/ops/<name>/op.hpp`, namespace `llaisys::ops`:

- `embedding(out[ntoken,hs], index[ntoken] i64, weight[voc,hs])`
- `rms_norm(out[ntoken,hs], in[ntoken,hs], weight[hs], eps)`
- `linear(out[ntoken,dout], in[ntoken,din], weight[dout,din], bias[dout]_or_nullptr)` — bias arg is `tensor_t`, pass `nullptr` for no bias (`attn_o_w` and all MLP linears have no bias; QKV projections do).
- `rope(out[seqlen,nhead,d], in[seqlen,nhead,d], pos_ids[seqlen] i64, theta)`
- `self_attention(attn_val[seqlen,nhead,dv], q[seqlen,nhead,d], k[total_len,nkvhead,d], v[total_len,nkvhead,dv], scale)` — causal, GQA-aware (`nhead` must be a multiple of `nkvhead`); internally `causal_offset = total_len - seqlen`.
- `swiglu(out[ntoken,di], gate[ntoken,di], up[ntoken,di])`
- `add(c, a, b)` — same shape, contiguous only
- `argmax(max_idx[1] i64, max_val[1], vals[N] 1D)` — `vals` must be 1D and contiguous, so slice logits down to just the last token's row before calling.

**`ops::rearrange(out, in)` is deprecated — a permanent `TO_BE_IMPLEMENTED()` stub, not just unfinished** (an Assignment #2 leftover) — don't call it, and don't propose implementing it. For copying newly-computed K/V into the KV cache, both sides are provably fully-contiguous (`Tensor::isContiguous()` treats a size-1 leading dim as contiguous regardless of its stride there), so a flat `std::memcpy` of `numel()*elementSize()` bytes is a correct, simpler substitute — see the reference file.

### KV cache design

`LlaisysQwen2Model::k_cache`/`v_cache` are `std::vector<llaisys::tensor_t>`, one entry per layer (`k_cache[l]` = layer `l`'s cache — each layer has independent K/V projection weights, so caches can't be shared across layers). Each is shaped `[maxseq, nkvh, dh]`. `model->cur_len` tracks how many positions are actually filled so far (independent of `maxseq`, which is just the preallocated capacity, persists across `Infer` calls on the same model instance).

Per `Infer` call: `total_len = cur_len + ntoken`; slice `[cur_len, total_len)` on dim 0 to write this step's new K/V, slice `[0, total_len)` to read the full history into `self_attention`. `meta.maxseq` should be capped well below the config's `max_position_embeddings` (e.g. 4096) in the Python wrapper — using the raw config value would allocate a multi-GB KV cache; this machine only has ~5.5GB free RAM.

### Local test model

`DeepSeek-R1-Distill-Qwen-1.5B/` (untracked) — `model.safetensors` is now the real ~3.5GB weights (the earlier git-lfs-pointer problem was resolved). `test/test_infer.py --model <dir> --test` is the acceptance test (greedy-decode token-for-token match against HF `transformers`); passes.

## Working style for this project

The user is doing these assignments to learn. **Do not generate finished implementations for the parts they're actively working through** — this now means the Assignment #4 CUDA work (`xmake/nvidia.lua`, `src/device/nvidia/nvidia_runtime_api.cu`, `src/ops/*/nvidia/*.cu`). Prefer step-by-step guidance: explain what a compiler error means, describe in words what the next piece of logic needs to do, answer conceptual questions directly, and let them write the actual code. Mechanical/environment-only fixes (wrong import path, wrong constant sourced from a config dict that's already being read for other fields) are fine to just fix and explain, rather than treating as "logic they're working through" — judgment call based on whether it's a design/algorithm decision vs. a one-line correction. The `docs/qwen2_*_reference.*` files exist for them to check against once stuck on Assignment #3 — don't paste their contents proactively; no CUDA reference files exist yet for Assignment #4.
