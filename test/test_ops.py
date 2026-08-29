import argparse
import subprocess
import sys
from pathlib import Path

OPS = [
    "add",
    "argmax",
    "embedding",
    "linear",
    "rms_norm",
    "rope",
    "self_attention",
    "swiglu",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    ops_dir = Path(__file__).parent / "ops"
    for op in OPS:
        script = ops_dir / f"{op}.py"
        print(f"=== {script} ===", flush=True)
        cmd = [sys.executable, str(script), "--device", args.device]
        if args.profile:
            cmd.append("--profile")
        rc = subprocess.call(cmd)
        if rc != 0:
            sys.exit(rc)

    print("\033[92mAll operator tests passed!\033[0m\n")


if __name__ == "__main__":
    main()
