#ifndef QWEN_ROPE_HPP
#define QWEN_ROPE_HPP

#include <cuda_runtime.h>
#include <cstdint>

namespace qwen {

// Applies Partial RoPE to Q or K tensors
// num_heads: total heads count
// head_dim: total dimension per head (e.g. 256)
// partial_rope_dim: dimensions to rotate (e.g. 64)
void rope_forward(
    float* q_or_k,
    int pos,
    int num_heads,
    int head_dim,
    int partial_rope_dim = 64,
    float rope_theta = 1000000.0f,
    cudaStream_t stream = nullptr
);

} // namespace qwen

#endif // QWEN_ROPE_HPP
