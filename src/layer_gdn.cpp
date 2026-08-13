#include "layer_gdn.hpp"
#include "gdn.hpp"
#include "ops.hpp"
#include "matmul_q4.hpp"
#include <iostream>

namespace qwen {

GdnLayer::~GdnLayer() {
    free_buffers();
}

void GdnLayer::free_buffers() {
    if (d_state_) { cudaFree(d_state_); d_state_ = nullptr; }
    if (d_in_proj_) { cudaFree(d_in_proj_); d_in_proj_ = nullptr; }
    if (d_gdn_out_) { cudaFree(d_gdn_out_); d_gdn_out_ = nullptr; }
    if (d_norm_out_) { cudaFree(d_norm_out_); d_norm_out_ = nullptr; }
    if (d_out_proj_) { cudaFree(d_out_proj_); d_out_proj_ = nullptr; }
}

void GdnLayer::init(const ModelConfig& cfg, int layer_idx) {
    cfg_ = cfg;
    layer_idx_ = layer_idx;
    free_buffers();

    size_t state_size = (size_t)cfg_.gdn_num_heads * cfg_.gdn_key_dim * cfg_.gdn_value_dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_state_, state_size));
    CUDA_CHECK(cudaMemset(d_state_, 0, state_size));

    int proj_dim = 2 * cfg_key_dim(cfg_) + cfg_value_dim(cfg_); // Q, K, V, Beta projections
    CUDA_CHECK(cudaMalloc(&d_in_proj_, proj_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gdn_out_, cfg_value_dim(cfg_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm_out_, cfg_value_dim(cfg_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_proj_, cfg_.d_model * sizeof(float)));
}

void GdnLayer::forward(
    const float* x_in,
    float* x_out,
    const QuantTensor& in_proj_w,
    const QuantTensor& out_proj_w,
    const float* norm_weight,
    const CudaContext& ctx
) {
    // 1. In-projection GEMM (W4A16)
    int proj_dim = 2 * cfg_key_dim(cfg_) + cfg_value_dim(cfg_);
    
    // Fallback zero/scale pointers if unquantized demo
    const float* dummy_scales = x_in; 
    const float* dummy_zeros = x_in;

    matmul_w4a16(
        static_cast<const uint8_t*>(in_proj_w.device_ptr),
        dummy_scales,
        dummy_zeros,
        x_in,
        d_in_proj_,
        proj_dim,
        cfg_.d_model,
        32,
        ctx.stream()
    );

    // 2. Split Q, K, V, B activation pointers
    const float* q_ptr = d_in_proj_;
    const float* k_ptr = d_in_proj_ + cfg_key_dim(cfg_);
    const float* v_ptr = d_in_proj_ + 2 * cfg_key_dim(cfg_);
    const float* b_ptr = d_in_proj_; // Shared key bias gate

    // 3. Recurrent Gated DeltaNet update
    gdn_recurrent_step(
        d_state_,
        q_ptr,
        k_ptr,
        v_ptr,
        b_ptr,
        d_gdn_out_,
        cfg_.gdn_num_heads,
        cfg_.gdn_key_dim,
        cfg_.gdn_value_dim,
        ctx.stream()
    );

    // 4. Gated RMSNorm
    rmsnorm_gated_forward(
        d_gdn_out_,
        q_ptr,
        norm_weight,
        d_norm_out_,
        cfg_value_dim(cfg_),
        1e-6f,
        ctx.stream()
    );

    // 5. Out-projection GEMM
    matmul_w4a16(
        static_cast<const uint8_t*>(out_proj_w.device_ptr),
        dummy_scales,
        dummy_zeros,
        d_norm_out_,
        d_out_proj_,
        cfg_.d_model,
        cfg_value_dim(cfg_),
        32,
        ctx.stream()
    );

    // 6. Residual Connection
    residual_add(x_in, d_out_proj_, x_out, cfg_.d_model, ctx.stream());
}

} // namespace qwen
