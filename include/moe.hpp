#ifndef QWEN_MOE_HPP
#define QWEN_MOE_HPP

#include <cuda_runtime.h>
#include <cstdint>

namespace qwen {

// Top-K Softmax kernel for MoE routing
// router_logits: (num_experts) logits from gate projection
// topk_indices: output array of top_k expert indices (length k)
// topk_weights: output array of normalized softmax weights (length k)
void moe_topk_softmax(
    const float* router_logits,
    int* topk_indices,
    float* topk_weights,
    int num_experts = 256,
    int k = 8,
    cudaStream_t stream = nullptr
);

// Accumulates expert outputs: out = sum_i (topk_weights[i] * expert_outputs[i])
// expert_outputs: k x d_model
void moe_expert_accumulate(
    const float* expert_outputs,
    const float* topk_weights,
    float* out,
    int k,
    int d_model,
    cudaStream_t stream = nullptr
);

} // namespace qwen

#endif // QWEN_MOE_HPP
