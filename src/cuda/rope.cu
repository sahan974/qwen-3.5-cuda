#include "rope.hpp"
#include <cmath>
#include <stdio.h>

namespace qwen {

__global__ void rope_kernel(
    float* __restrict__ q_or_k,
    int pos,
    int num_heads,
    int head_dim,
    int partial_rope_dim,
    float rope_theta
) {
    int head_idx = blockIdx.x;
    int pair_idx = threadIdx.x; // handles pairs (0,1), (2,3) ... up to partial_rope_dim / 2

    if (head_idx >= num_heads || pair_idx >= (partial_rope_dim / 2)) return;

    int i0 = head_idx * head_dim + pair_idx * 2;
    int i1 = i0 + 1;

    float theta = static_cast<float>(pos) / powf(rope_theta, static_cast<float>(pair_idx * 2) / static_cast<float>(partial_rope_dim));
    float cos_th = cosf(theta);
    float sin_th = sinf(theta);

    float v0 = q_or_k[i0];
    float v1 = q_or_k[i1];

    q_or_k[i0] = v0 * cos_th - v1 * sin_th;
    q_or_k[i1] = v0 * sin_th + v1 * cos_th;
}

void rope_forward(
    float* q_or_k,
    int pos,
    int num_heads,
    int head_dim,
    int partial_rope_dim,
    float rope_theta,
    cudaStream_t stream
) {
    int pairs = partial_rope_dim / 2;
    int threads = (pairs < 32) ? 32 : pairs;
    dim3 grid(num_heads);

    rope_kernel<<<grid, threads, 0, stream>>>(
        q_or_k, pos, num_heads, head_dim, partial_rope_dim, rope_theta
    );
}

} // namespace qwen
