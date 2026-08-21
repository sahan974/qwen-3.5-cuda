#include "layer_moe.hpp"
#include "matmul_q4.hpp"
#include <cmath>
#include <stdexcept>

namespace qwen {

// --- External CUDA Kernel Declarations ---
void moe_topk(const float* router, int* top_idx, float* top_weight, int experts, int k, cudaStream_t stream);
void moe_silu_mul(const float* gate, const float* up, float* hidden, int dim, cudaStream_t stream);
void moe_add(float* accum, const float* expert_out, float weight, int dim, cudaStream_t stream);
void moe_shared_add(float* accum, const float* shared_out, const float* sgate, int dim, cudaStream_t stream);


// --- Lifecycle ---
MoeLayer::~MoeLayer() {
    free_buffers();
}

void MoeLayer::free_buffers() {
    float** float_buffers[] = {
        &router_, &gate_, &up_, &hidden_, &expert_out_, &accum_,
        &sgate_, &shared_gate_, &shared_up_, &shared_hidden_, &shared_out_
    };
    for (float** p : float_buffers) {
        if (*p) {
            cudaFree(*p);
            *p = nullptr;
        }
    }

    if (top_idx_) {
        cudaFree(top_idx_);
        top_idx_ = nullptr;
    }
    if (top_weight_) {
        cudaFree(top_weight_);
        top_weight_ = nullptr;
    }

    if (h_idx_) {
        cudaFreeHost(h_idx_);
        h_idx_ = nullptr;
    }
    if (h_weight_) {
        cudaFreeHost(h_weight_);
        h_weight_ = nullptr;
    }
}

void MoeLayer::init(const ModelConfig& c, int li) {
    free_buffers();
    cfg_ = c;
    layer_idx_ = li;

    // Allocate Routing Buffers
    CUDA_CHECK(cudaMalloc(&router_,     c.num_experts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&top_idx_,    c.num_experts_per_tok * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&top_weight_, c.num_experts_per_tok * sizeof(float)));

    // Allocate Sparse Expert Buffers
    CUDA_CHECK(cudaMalloc(&gate_,       c.moe_intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&up_,         c.moe_intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hidden_,     c.moe_intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&expert_out_, c.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&accum_,      c.d_model * sizeof(float)));

    // Allocate Shared Expert Buffers
    CUDA_CHECK(cudaMalloc(&sgate_,         sizeof(float)));
    CUDA_CHECK(cudaMalloc(&shared_gate_,   c.shared_expert_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&shared_up_,     c.shared_expert_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&shared_hidden_, c.shared_expert_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&shared_out_,    c.d_model * sizeof(float)));

    // Allocate Host-side Routing Buffers (Pinned Memory)
    CUDA_CHECK(cudaHostAlloc(&h_idx_,    c.num_experts_per_tok * sizeof(int),   cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_weight_, c.num_experts_per_tok * sizeof(float), cudaHostAllocDefault));
}


// --- Forward Pass ---
void MoeLayer::forward(
    const float*       x,
    float*             out,
    const QuantTensor& r,
    const QuantTensor& eg,
    const QuantTensor& eu,
    const QuantTensor& ed,
    const QuantTensor& sgo,
    const QuantTensor& sg,
    const QuantTensor& su,
    const QuantTensor& sd,
    const CudaContext& ctx
) {
    cudaStream_t stream = ctx.stream();

    // 1. Compute MoE Routing
    matmul_dispatch(r, x, router_, cfg_.num_experts, cfg_.d_model, stream);
    moe_topk(router_, top_idx_, top_weight_, cfg_.num_experts, cfg_.num_experts_per_tok, stream);

    // 2. Fetch routing choices to CPU
    CUDA_CHECK(cudaMemcpyAsync(h_idx_,    top_idx_,    cfg_.num_experts_per_tok * sizeof(int),   cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_weight_, top_weight_, cfg_.num_experts_per_tok * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // Synchronize to ensure CPU has the indices before proceeding
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemsetAsync(accum_, 0, cfg_.d_model * sizeof(float), stream));

    // Tensor offsets for sliced batched execution
    uint64_t gu_stride   = (uint64_t)cfg_.d_model * cfg_.moe_intermediate_dim;
    uint64_t down_stride = (uint64_t)cfg_.moe_intermediate_dim * cfg_.d_model;

    // 3. Validate Routing Weights
    float route_sum = 0.0f;
    for (int j = 0; j < cfg_.num_experts_per_tok; ++j) {
        if (h_idx_[j] < 0 || h_idx_[j] >= cfg_.num_experts || !std::isfinite(h_weight_[j]) || h_weight_[j] < 0.0f) {
            throw std::runtime_error("invalid/non-finite MoE routing result in layer " + std::to_string(layer_idx_));
        }
        route_sum += h_weight_[j];
    }

    if (!std::isfinite(route_sum) || std::fabs(route_sum - 1.0f) > 1e-4f) {
        throw std::runtime_error("MoE routing weights are not normalized in layer " + std::to_string(layer_idx_));
    }

    // 4. Compute Sparse Experts Iteratively
    for (int j = 0; j < cfg_.num_experts_per_tok; ++j) {
        int e = h_idx_[j];

        // Expert Projections
        matmul_dispatch(eg, x, gate_, cfg_.moe_intermediate_dim, cfg_.d_model, stream, e * gu_stride);
        matmul_dispatch(eu, x, up_,   cfg_.moe_intermediate_dim, cfg_.d_model, stream, e * gu_stride);

        // Expert SiLU + Mul
        moe_silu_mul(gate_, up_, hidden_, cfg_.moe_intermediate_dim, stream);

        // Expert Down Projection
        matmul_dispatch(ed, hidden_, expert_out_, cfg_.d_model, cfg_.moe_intermediate_dim, stream, e * down_stride);

        // Accumulate Weighted Output
        moe_add(accum_, expert_out_, h_weight_[j], cfg_.d_model, stream);
    }

    // 5. Compute Shared Expert
    matmul_dispatch(sgo, x, sgate_,       1,                     cfg_.d_model, stream);
    matmul_dispatch(sg,  x, shared_gate_, cfg_.shared_expert_dim, cfg_.d_model, stream);
    matmul_dispatch(su,  x, shared_up_,   cfg_.shared_expert_dim, cfg_.d_model, stream);

    moe_silu_mul(shared_gate_, shared_up_, shared_hidden_, cfg_.shared_expert_dim, stream);
    matmul_dispatch(sd, shared_hidden_, shared_out_, cfg_.d_model, cfg_.shared_expert_dim, stream);

    // 6. Accumulate Shared Expert and Output
    moe_shared_add(accum_, shared_out_, sgate_, cfg_.d_model, stream);
    CUDA_CHECK(cudaMemcpyAsync(out, accum_, cfg_.d_model * sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

} // namespace qwen
