#include "layer_moe.hpp"
#include "moe.hpp"
#include "ops.hpp"
#include "matmul_q4.hpp"
#include <iostream>

namespace qwen {

MoeLayer::~MoeLayer() {
    free_buffers();
}

void MoeLayer::free_buffers() {
    if (d_router_logits_) { cudaFree(d_router_logits_); d_router_logits_ = nullptr; }
    if (d_topk_indices_) { cudaFree(d_topk_indices_); d_topk_indices_ = nullptr; }
    if (d_topk_weights_) { cudaFree(d_topk_weights_); d_topk_weights_ = nullptr; }
    if (d_expert_outputs_) { cudaFree(d_expert_outputs_); d_expert_outputs_ = nullptr; }
    if (d_moe_out_) { cudaFree(d_moe_out_); d_moe_out_ = nullptr; }
    if (d_shared_out_) { cudaFree(d_shared_out_); d_shared_out_ = nullptr; }
}

void MoeLayer::init(const ModelConfig& cfg, int layer_idx) {
    cfg_ = cfg;
    layer_idx_ = layer_idx;
    free_buffers();

    CUDA_CHECK(cudaMalloc(&d_router_logits_, cfg_.num_experts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk_indices_, cfg_.num_experts_per_tok * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_topk_weights_, cfg_.num_experts_per_tok * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_expert_outputs_, (size_t)cfg_.num_experts_per_tok * cfg_.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_moe_out_, cfg_.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_shared_out_, cfg_.d_model * sizeof(float)));
}

void MoeLayer::forward(
    const float* x_in,
    float* x_out,
    const QuantTensor& router_w,
    const QuantTensor& shared_expert_w,
    const CudaContext& ctx
) {
    const float* dummy_scales = x_in;
    const float* dummy_zeros = x_in;

    // 1. Router GEMM: (2048 -> 256)
    matmul_w4a16(
        static_cast<const uint8_t*>(router_w.device_ptr),
        dummy_scales,
        dummy_zeros,
        x_in,
        d_router_logits_,
        cfg_.num_experts,
        cfg_.d_model,
        32,
        ctx.stream()
    );

    // 2. Top-K Softmax selection
    moe_topk_softmax(
        d_router_logits_,
        d_topk_indices_,
        d_topk_weights_,
        cfg_.num_experts,
        cfg_.num_experts_per_tok,
        ctx.stream()
    );

    // 3. Accumulate Expert Outputs
    moe_expert_accumulate(
        d_expert_outputs_,
        d_topk_weights_,
        d_moe_out_,
        cfg_.num_experts_per_tok,
        cfg_.d_model,
        ctx.stream()
    );

    // 4. Shared Expert Forward GEMM
    matmul_w4a16(
        static_cast<const uint8_t*>(shared_expert_w.device_ptr),
        dummy_scales,
        dummy_zeros,
        x_in,
        d_shared_out_,
        cfg_.d_model,
        cfg_.d_model,
        32,
        ctx.stream()
    );

    // 5. Combine Experts + Shared Expert + Residual Add
    residual_add(d_moe_out_, d_shared_out_, d_moe_out_, cfg_.d_model, ctx.stream());
    residual_add(x_in, d_moe_out_, x_out, cfg_.d_model, ctx.stream());
}

} // namespace qwen
