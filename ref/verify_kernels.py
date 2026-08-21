#!/usr/bin/env python3
"""
Run compiled CUDA checks and, optionally, a real-model generation smoke test.
"""
import argparse
import pathlib
import subprocess
import sys


def run(cmd):
    """Executes a command, prints it, and captures output."""
    print("+", " ".join(map(str, cmd)), flush=True)

    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    if result.returncode:
        # Print output and exit on failure
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        raise SystemExit(f"Command failed with exit code {result.returncode}")

    return result.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=pathlib.Path, default=pathlib.Path("build"))
    ap.add_argument("--weights", type=pathlib.Path)
    args = ap.parse_args()

    # Determine executable extension based on OS
    suffix = ".exe" if sys.platform == "win32" else ""
    tests = args.build / ("qwen-kernel-tests" + suffix)
    engine = args.build / ("qwen-3.5-cuda" + suffix)

    if not tests.exists():
        raise SystemExit(f"Missing {tests}; configure and build the project first.")

    # 1. Run the base CUDA kernel unit tests
    print(run([tests]))

    if args.weights:
        if not args.weights.is_file():
            raise SystemExit(f"Missing model weights file: {args.weights}")

        # 2. Run Tokenizer Test
        tokenized = run([
            engine,
            "--weights", args.weights,
            "--prompt", "The capital of France is",
            "--tokenize-only"
        ])
        print(tokenized)

        if "TOKEN_IDS: 760 6511 314 9338 369" not in tokenized:
            raise SystemExit("Tokenizer check failed: Prompt IDs differ from the Qwen reference.")

        # 3. Run End-to-End Generation Test
        output = run([
            engine,
            "--weights", args.weights,
            "--prompt", "The capital of France is",
            "--max", "8"
        ])
        print(output)

        if "paris" not in output.lower():
            raise SystemExit("End-to-end check failed: Generated output did not contain 'Paris'.")

    else:
        print("\nCUDA kernels passed. End-to-end generation was not run (no --weights supplied).")


if __name__ == "__main__":
    main()
