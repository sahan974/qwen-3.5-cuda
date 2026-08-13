# qwen-3.5-cuda

A correctness-first C++17/CUDA inference engine for the text stack of
Qwen3.5-35B-A3B GGUF models on a single CUDA GPU.

The engine implements the Qwen3.5 hybrid decoder rather than treating its
linear-attention layers as Mamba/SSM layers:

- 30 recurrent Gated DeltaNet layers with causal depthwise convolution,
  Q/K L2 normalization, learned decay, delta-rule state updates, and gated
  per-head RMSNorm;
- 10 gated grouped-query full-attention layers with joint query/gate
  projections, Q/K RMSNorm, partial split-half RoPE, KV caching, and sigmoid
  output gates;
- routed top-8 MoE plus the sigmoid-gated shared expert on every layer;
- GGML F32/F16/BF16, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, and Q6_K weight layouts;
- GGUF byte-level BPE tokens and merges, prompt prefill, EOS handling, and a
  greedy autoregressive decode loop.

Missing tensors, wrong shapes, unsupported quantization types, invalid token
IDs, discontinuous cache positions, and non-finite logits are fatal errors.
There are no zero-output, pass-through, or fake dry-run fallbacks.

## Build

CUDA Toolkit 12.x and CMake 3.25+ are required.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Use architecture `86` for RTX 3090, `89` for RTX 4090, or `80` for A100.

## Verify

The verifier executes compiled CUDA kernels; it does not merely calculate a
separate reference value.

```bash
python3 ref/verify_kernels.py --build build
python3 ref/verify_kernels.py --build build \
  --weights /workspace/Qwen3.5-35B-A3B-Q4_K_M.gguf
```

The first command checks Q4_K/Q5_K/Q6_K block layouts and the recurrent
DeltaNet update. The second first checks that the prompt tokenizes to the known
Qwen IDs `[760, 6511, 314, 9338, 369]`, then runs real end-to-end inference and
fails unless the generated answer contains `Paris`.

## Run

```bash
./build/qwen-3.5-cuda \
  --weights /workspace/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --prompt "The capital of France is" \
  --max 16 \
  --ctx 4096
```

The model plus runtime caches must fit in VRAM. A Q4_K_M 35B-A3B GGUF is close
to the practical limit of a 24 GB card, so close other GPU workloads first.

## Correctness references

The layer graph follows the official Qwen3.5 implementation in Hugging Face
Transformers and the Qwen3.5-MoE GGUF graph in llama.cpp. GGML quantized block
decoding follows llama.cpp's `ggml-quants.c` layouts.
