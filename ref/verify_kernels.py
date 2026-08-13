#!/usr/bin/env python3
"""Run compiled CUDA checks and, optionally, a real-model generation smoke test."""
import argparse
import pathlib
import subprocess
import sys

def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=pathlib.Path, default=pathlib.Path("build"))
    ap.add_argument("--weights", type=pathlib.Path)
    args = ap.parse_args()
    suffix = ".exe" if sys.platform == "win32" else ""
    tests = args.build / ("qwen-kernel-tests" + suffix)
    engine = args.build / ("qwen-3.5-cuda" + suffix)
    if not tests.exists():
        raise SystemExit(f"missing {tests}; configure and build the project first")
    print(run([tests]))
    if args.weights:
        if not args.weights.is_file(): raise SystemExit(f"missing model: {args.weights}")
        tokenized = run([engine, "--weights", args.weights, "--prompt", "The capital of France is", "--tokenize-only"])
        print(tokenized)
        if "TOKEN_IDS: 760 6511 314 9338 369" not in tokenized:
            raise SystemExit("tokenizer check failed: prompt IDs differ from the Qwen reference")
        output = run([engine, "--weights", args.weights, "--prompt", "The capital of France is", "--max", "8"])
        print(output)
        if "paris" not in output.lower():
            raise SystemExit("end-to-end check failed: generated output did not contain 'Paris'")
    else:
        print("CUDA kernels passed. End-to-end generation was not run (no --weights supplied).")

if __name__ == "__main__":
    main()
