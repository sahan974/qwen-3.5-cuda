#ifndef QWEN_OPS_HPP
#define QWEN_OPS_HPP

#include <cuda_runtime.h>
#include <cstdint>

namespace qwen {

// GGUF ordinary RMSNorm weights are already 1-centered by the converter.
// out = norm(x) * weight
void rmsnorm_forward(
    const float* x,
    const float* weight,
    float* out,
    int N,
    float eps = 1e-6f,
    cudaStream_t stream = nullptr
);

// Gated RMSNorm: out = norm(x) * weight * silu(gate)
void rmsnorm_gated_forward(
    const float* x,
    const float* gate,
    const float* weight,
    float* out,
    int N,
    float eps = 1e-6f,
    cudaStream_t stream = nullptr
);

// Residual add: out = x1 + x2
void residual_add(
    const float* x1,
    const float* x2,
    float* out,
    int size,
    cudaStream_t stream = nullptr
);

// Token embedding lookup: out = embed_table[token_id]
void encoder_forward(
    int token_id,
    const float* embed_table,
    float* out,
    int d_model,
    cudaStream_t stream = nullptr
);

} // namespace qwen

#endif // QWEN_OPS_HPP
