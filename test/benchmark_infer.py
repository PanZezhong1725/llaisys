import argparse
import contextlib
import time

from test_utils import llaisys_device
import llaisys
from transformers import AutoTokenizer


def make_step_context(record, api, use_nvtx):
    @contextlib.contextmanager
    def step_context(step, is_prefill):
        if use_nvtx:
            import torch

            torch.cuda.nvtx.range_push(f"{'prefill' if is_prefill else 'decode'}_{step}")
        start = time.perf_counter()
        try:
            yield
        finally:
            api.device_synchronize()
            record.append((is_prefill, time.perf_counter() - start))
            if use_nvtx:
                import torch

                torch.cuda.nvtx.range_pop()

    return step_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "iluvatar"], type=str)
    parser.add_argument("--prompt", default="Who are you?", type=str)
    parser.add_argument("--max_steps", default=128, type=int)
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="wrap each prefill/decode step in an NVTX range for nsys (needs torch + an NVIDIA device)",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)

    model = llaisys.models.Qwen2(args.model, llaisys_device(args.device))
    api = llaisys.RuntimeAPI(llaisys_device(args.device))

    record = []
    step_context = make_step_context(record, api, args.nvtx)

    start = time.perf_counter()
    outputs = model.generate(inputs, max_new_tokens=args.max_steps, step_context=step_context)
    total_elapsed = time.perf_counter() - start

    prefill = [elapsed for is_prefill, elapsed in record if is_prefill]
    decode = [elapsed for is_prefill, elapsed in record if not is_prefill]

    print(f"Prompt tokens: {len(inputs)}, generated: {len(outputs) - len(inputs)}")
    print(f"Prefill: {prefill[0] * 1000:.2f} ms")
    if decode:
        avg_decode = sum(decode) / len(decode)
        print(f"Decode: {len(decode)} steps, avg {avg_decode * 1000:.2f} ms/token, {1.0 / avg_decode:.2f} tokens/s")
    print(f"Total: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
