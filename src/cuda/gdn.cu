#include "gdn.hpp"
#include <cmath>
#include <stdio.h>

namespace qwen {

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Kernel: Each threadblock handles one head (num_heads total grid size)
__global__ void gdn_recurrent_kernel(
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ b,
    float* __restrict__ out,
    int key_dim,
    int value_dim
) {
    int head_idx = blockIdx.x;
    int v_idx = threadIdx.x; // Handles value_dim elements in parallel

    if (v_idx >= value_dim) return;

    size_t head_state_offset = (size_t)head_idx * key_dim * value_dim;
    float* head_state = state + head_state_offset;

    const float* head_q = q + head_idx * key_dim;
    const float* head_k = k + head_idx * key_dim;
    const float* head_v = v + head_idx * value_dim;
    const float* head_b = b + head_idx * key_dim;

    float v_t = head_v[v_idx];
    float out_val = 0.0f;

    // Recurrent update over key_dim dimension
    for (int k_idx = 0; k_idx < key_dim; ++k_idx) {
        float beta = sigmoid(head_b[k_idx]);
        float k_t = head_k[k_idx];
        float q_t = head_q[k_idx];

        // S_t(k, v) = (1 - beta * k_t) * S_{t-1}(k, v) + beta * v_t
        size_t state_idx = (size_t)k_idx * value_dim + v_idx;
        float prev_s = head_state[state_idx];
        float new_s = (1.0f - beta * k_t) * prev_s + beta * v_t;

        head_state[state_idx] = new_s;

        // Output calculation: out(v) = sum_k (S_t(k, v) * q(k))
        out_val += new_s * q_t;
    }

    out[head_idx * value_dim + v_idx] = out_val;
}

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
    cudaStream_t stream
) {
    int threads = (value_dim < 256) ? value_dim : 256;
    dim3 grid(num_heads);

    gdn_recurrent_kernel<<<grid, threads, 0, stream>>>(
        state, q, k, v, b, out, key_dim, value_dim
    );
}

} // namespace qwen
