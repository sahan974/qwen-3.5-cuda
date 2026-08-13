#include "layer_attn.hpp"
#include "ops.hpp"
#include "rope.hpp"
#include "matmul_q4.hpp"
#include <iostream>

namespace qwen {

AttnLayer::~AttnLayer() {
    free_buffers();
}

void AttnLayer::free_buffers() {
    if (d_k_cache_) { cudaFree(d_k_cache_); d_k_cache_ = nullptr; }
    if (d_v_cache_) { cudaFree(d_v_cache_); d_v_cache_ = nullptr; }
    if (d_q_) { cudaFree(d_q_); d_q_ = nullptr; }
    if (d_k_) { cudaFree(d_k_); d_k_ = nullptr; }
    if (d_v_) { cudaFree(d_v_); d_v_ = nullptr; }
    if (d_attn_out_) { cudaFree(d_attn_out_); d_attn_out_ = nullptr; }
    if (d_out_proj_) { cudaFree(d_out_proj_); d_out_proj_ = nullptr; }
}

void AttnLayer::init(const ModelConfig& cfg, int layer_idx, int max_seq_len) {
    cfg_ = cfg;
    layer_idx_ = layer_idx;
    max_seq_len_ = max_seq_len;
    free_buffers();

    size_t kv_size = (size_t)max_seq_len_ * cfg_.num_kv_heads * cfg_.head_dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_k_cache_, kv_size));
    CUDA_CHECK(cudaMalloc(&d_v_cache_, kv_size));
    CUDA_CHECK(cudaMemset(d_k_cache_, 0, kv_size));
    CUDA_CHECK(cudaMemset(d_v_cache_, 0, kv_size));

    size_t q_size = (size_t)cfg_.num_heads * cfg_.head_dim * sizeof(float);
    size_t k_size = (size_t)cfg_.num_kv_heads * cfg_.head_dim * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_q_, q_size));
    CUDA_CHECK(cudaMalloc(&d_k_, k_size));
    CUDA_CHECK(cudaMalloc(&d_v_, k_size));
    CUDA_CHECK(cudaMalloc(&d_attn_out_, q_size));
    CUDA_CHECK(cudaMalloc(&d_out_proj_, cfg_.d_model * sizeof(float)));
}

void AttnLayer::forward(
    const float* x_in,
    float* x_out,
    int pos,
    const QuantTensor& q_proj_w,
    const QuantTensor& k_proj_w,
    const QuantTensor& v_proj_w,
    const QuantTensor& out_proj_w,
    const float* q_norm_w,
    const float* k_norm_w,
    const CudaContext& ctx
) {
    const float* dummy_scales = x_in;
    const float* dummy_zeros = x_in;

    // 1. Projections
    matmul_w4a16(static_cast<const uint8_t*>(q_proj_w.device_ptr), dummy_scales, dummy_zeros, x_in, d_q_, cfg_.num_heads * cfg_.head_dim, cfg_.d_model, 32, ctx.stream());
    matmul_w4a16(static_cast<const uint8_t*>(k_proj_w.device_ptr), dummy_scales, dummy_zeros, x_in, d_k_, cfg_.num_kv_heads * cfg_.head_dim, cfg_.d_model, 32, ctx.stream());
    matmul_w4a16(static_cast<const uint8_t*>(v_proj_w.device_ptr), dummy_scales, dummy_zeros, x_in, d_v_, cfg_.num_kv_heads * cfg_.head_dim, cfg_.d_model, 32, ctx.stream());

    // 2. Q and K pre-norm
    rmsnorm_forward(d_q_, q_norm_w, d_q_, cfg_.num_heads * cfg_.head_dim, 1e-6f, ctx.stream());
    rmsnorm_forward(d_k_, k_norm_w, d_k_, cfg_.num_kv_heads * cfg_.head_dim, 1e-6f, ctx.stream());

    // 3. Partial RoPE (64 dims)
    rope_forward(d_q_, pos, cfg_.num_heads, cfg_.head_dim, cfg_.partial_rope_dim, 1000000.0f, ctx.stream());
    rope_forward(d_k_, pos, cfg_.num_kv_heads, cfg_.head_dim, cfg_.partial_rope_dim, 1000000.0f, ctx.stream());

    // 4. Update KV cache
    size_t kv_head_bytes = (size_t)cfg_.num_kv_heads * cfg_.head_dim * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(d_k_cache_ + (size_t)pos * cfg_.num_kv_heads * cfg_.head_dim, d_k_, kv_head_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(d_v_cache_ + (size_t)pos * cfg_.num_kv_heads * cfg_.head_dim, d_v_, kv_head_bytes, cudaMemcpyDeviceToDevice, ctx.stream()));

    // 5. Out projection GEMM
    matmul_w4a16(static_cast<const uint8_t*>(out_proj_w.device_ptr), dummy_scales, dummy_zeros, d_q_, d_out_proj_, cfg_.d_model, cfg_.num_heads * cfg_.head_dim, 32, ctx.stream());

    // 6. Residual Add
    residual_add(x_in, d_out_proj_, x_out, cfg_.d_model, ctx.stream());
}

} // namespace qwen
