#ifndef QWEN_GDN_HPP
#define QWEN_GDN_HPP

#include <cuda_runtime.h>
#include <cstdint>

namespace qwen {

// Gated DeltaNet Recurrent Update Kernel
// state: (num_heads, key_dim, value_dim) persistent state tensor
// q: (num_heads, key_dim)
// k: (num_heads, key_dim)
// v: (num_heads, value_dim)
// b: (num_heads, key_dim) beta gate values
// out: (num_heads, value_dim) output activation
void gdn_recurrent_step(
    float* state,
    const float* q,
    const float* k,
    const float* v,
    const float* b,
    float* out,
    int num_heads,
    int key_dim,
    int value_dim,
    cudaStream_t stream = nullptr
);

} // namespace qwen

#endif // QWEN_GDN_HPP
