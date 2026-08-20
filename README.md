# qwen-3.5-cuda

A C++17 and CUDA inference engine for Qwen3.5-35B-A3B GGUF models.

The runtime implements the Qwen3.5 hybrid text decoder directly in CUDA and
loads supported GGUF weights without converting them to a separate runtime
format.

## Features

- Recurrent Gated DeltaNet layers with causal depthwise convolution, Q/K L2
  normalization, learned decay, delta-rule state updates, and gated per-head
  RMSNorm
- Gated grouped-query attention with Q/K RMSNorm, partial split-half RoPE,
  causal attention, and KV caching
- Top-k routed MoE layers with normalized routing weights and a sigmoid-gated
  shared expert
- F32, F16, BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, and Q6_K GGML weight layouts
- Qwen3.5 Unicode pre-tokenization, byte-level BPE merges, and atomic GGUF
  special-token handling
- Raw text completion and ChatML-formatted chat prompts
- Greedy decoding and configurable temperature, top-k, top-p, repetition
  penalty, history window, and deterministic seed
- Explicit EOG and maximum-token stop reporting
- GGUF metadata, tensor shape, data range, alignment, and overlap validation
- Fail-fast handling for missing tensors, unsupported layouts, invalid routing,
  invalid token IDs, cache-position errors, and non-finite values
- Optional MTP tensors are detected and excluded from base-model inference

## Requirements

- CMake 3.25 or newer
- A C++17 compiler
- NVIDIA CUDA Toolkit 12.x or newer
- An NVIDIA GPU with enough VRAM for the selected GGUF and runtime caches

## Build

Configure the CUDA architecture for the target GPU. For an RTX 3090, use
architecture `86`:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_TESTING=ON

cmake --build build -j"$(nproc)"
```

Common CUDA architectures:

| GPU | Architecture |
| --- | ---: |
| A100 | 80 |
| RTX 3090 | 86 |
| RTX 4090 | 89 |

## Model

Pass the GGUF file at runtime with `--weights`. The commands below use:

```text
/workspace/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf
```

Confirm that the file is available:

```bash
find /workspace -type f -iname '*.gguf' -printf '%p  %s bytes\n'
```

## Raw completion

Temperature `0` performs deterministic greedy decoding:

```bash
CUDA_VISIBLE_DEVICES=0 ./build/qwen-3.5-cuda \
  --weights /workspace/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --prompt "The capital of France is" \
  --max 64 \
  --ctx 4096 \
  --temperature 0
```

## Chat

Use `--chat` to format the prompt using the ChatML tokens declared by the GGUF
tokenizer:

```bash
CUDA_VISIBLE_DEVICES=0 ./build/qwen-3.5-cuda \
  --weights /workspace/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --chat \
  --system "You are a concise, accurate assistant." \
  --prompt "Explain why the sky appears blue in three sentences." \
  --max 512 \
  --ctx 4096 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --repeat-penalty 1.05 \
  --repeat-last-n 64 \
  --seed 42
```

Generation ends with an explicit reason:

```text
[stop: eog]
```

or:

```text
[stop: max-tokens]
```

## Tokenizer check

Inspect prompt token IDs without loading model tensors into GPU memory:

```bash
./build/qwen-3.5-cuda \
  --weights /workspace/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --prompt "The capital of France is" \
  --tokenize-only
```

For this prompt, the expected token IDs are:

```text
760 6511 314 9338 369
```

Chat tokenization can be checked separately:

```bash
./build/qwen-3.5-cuda \
  --weights /workspace/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --chat \
  --prompt "What is the capital of France?" \
  --tokenize-only
```

## Tests

Run the CTest suite after building:

```bash
ctest --test-dir build --output-on-failure
```

The tests cover:

- Sampling filters, repetition penalties, deterministic seeding, and stopping
- Supported quantized block layouts and dequantization
- Two-step tiled GDN recurrence
- RMSNorm and Q/K normalization
- Partial split-half RoPE
- Grouped-query attention and cache behavior
- MoE routing and expert computation

Run the end-to-end verification helper with a model file:

```bash
CUDA_VISIBLE_DEVICES=0 python3 ref/verify_kernels.py \
  --build build \
  --weights /workspace/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf
```

## Benchmark

Measured on one NVIDIA RTX 3090 with the Qwen3.5-35B-A3B Q4_K_M GGUF, a
36-token prompt, 128 generated tokens, a 4096-token context, and greedy
decoding. The table reports the median of five runs.

| Metric | Median |
| --- | ---: |
| Model load time | 4.837 s |
| Prompt prefill throughput | 22.301 tok/s |
| Time to first token | 1.615 s |
| Generation throughput | 22.215 tok/s |
| Total inference time | 7.378 s |
| Observed runtime VRAM allocation | 20,774 MiB |

Run the same benchmark with:

```bash
CUDA_VISIBLE_DEVICES=0 ./build/qwen-3.5-cuda \
  --weights /workspace/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --prompt-file /workspace/benchmark-prompt.txt \
  --max 128 \
  --ctx 4096 \
  --temperature 0 \
  --benchmark
```

## Command-line options

| Option | Description |
| --- | --- |
| `--weights PATH` | GGUF model path |
| `--prompt TEXT` | User prompt or raw completion prefix |
| `--prompt-file PATH` | Read the prompt from a file |
| `--chat` | Apply ChatML prompt formatting |
| `--system TEXT` | System message used with `--chat` |
| `--max N` | Maximum number of generated tokens |
| `--ctx N` | Context capacity |
| `--temperature T` | Sampling temperature; `0` selects greedy decoding |
| `--top-k K` | Keep the K highest-logit candidates |
| `--top-p P` | Keep the smallest candidate set reaching probability P |
| `--repeat-penalty P` | Penalize tokens appearing in recent history |
| `--repeat-last-n N` | Number of recent tokens used by repetition penalty |
| `--seed N` | Sampling random seed |
| `--benchmark` | Print detailed load and inference measurements |
| `--tokenize-only` | Print token IDs without running inference |

## Project layout

| Path | Purpose |
| --- | --- |
| `include/` | Runtime interfaces and model data structures |
| `src/` | GGUF loading, tokenization, model orchestration, and sampling |
| `src/cuda/` | CUDA kernels for matrix operations, GDN, attention, RoPE, and MoE |
| `tests/` | CPU and CUDA correctness tests |
| `ref/` | Verification helpers |
| `third_party/llama.cpp/` | Vendored Unicode data used by the tokenizer |

## References

The vendored Unicode data retains its upstream MIT license notice.
