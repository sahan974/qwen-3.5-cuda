#include "moe.hpp"
#include <cmath>
#include <cfloat>
#include <stdio.h>

namespace qwen {

// Top-K Softmax Kernel (handles top_k up to 8 from up to 256 experts)
__global__ void moe_topk_softmax_kernel(
    const float* __restrict__ logits,
    int* __restrict__ topk_indices,
    float* __restrict__ topk_weights,
    int num_experts,
    int k
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Local array tracking top-k indices and values
    float top_vals[16];
    int top_inds[16];

    for (int i = 0; i < k; ++i) {
        top_vals[i] = -FLT_MAX;
        top_inds[i] = -1;
    }

    for (int i = 0; i < num_experts; ++i) {
        float val = logits[i];
        if (val > top_vals[k - 1]) {
            top_vals[k - 1] = val;
            top_inds[k - 1] = i;

            // Insertion sort to maintain descending order
            for (int j = k - 1; j > 0; --j) {
                if (top_vals[j] > top_vals[j - 1]) {
                    float temp_v = top_vals[j];
                    top_vals[j] = top_vals[j - 1];
                    top_vals[j - 1] = temp_v;

                    int temp_i = top_inds[j];
                    top_inds[j] = top_inds[j - 1];
                    top_inds[j - 1] = temp_i;
                }
            }
        }
    }

    // Compute max for numerical stability softmax
    float max_val = top_vals[0];
    float sum_exp = 0.0f;

    for (int i = 0; i < k; ++i) {
        top_vals[i] = expf(top_vals[i] - max_val);
        sum_exp += top_vals[i];
    }

    // Write back top-k indices and normalized weights
    for (int i = 0; i < k; ++i) {
        topk_indices[i] = top_inds[i];
        topk_weights[i] = top_vals[i] / sum_exp;
    }
}

// Expert Accumulate Kernel: out = sum_i (weights[i] * expert_outputs[i])
__global__ void moe_expert_accumulate_kernel(
    const float* __restrict__ expert_outputs,
    const float* __restrict__ topk_weights,
    float* __restrict__ out,
    int k,
    int d_model
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_model) return;

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        float weight = topk_weights[i];
        float val = expert_outputs[(size_t)i * d_model + idx];
        sum += weight * val;
    }

    out[idx] = sum;
}

void moe_topk_softmax(
    const float* router_logits,
    int* topk_indices,
    float* topk_weights,
    int num_experts,
    int k,
    cudaStream_t stream
) {
    moe_topk_softmax_kernel<<<1, 1, 0, stream>>>(
        router_logits, topk_indices, topk_weights, num_experts, k
    );
}

void moe_expert_accumulate(
    const float* expert_outputs,
    const float* topk_weights,
    float* out,
    int k,
    int d_model,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (d_model + threads - 1) / threads;
    moe_expert_accumulate_kernel<<<blocks, threads, 0, stream>>>(
        expert_outputs, topk_weights, out, k, d_model
    );
}

} // namespace qwen
