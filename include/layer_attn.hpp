#ifndef QWEN_LAYER_ATTN_HPP
#define QWEN_LAYER_ATTN_HPP

#include "config.hpp"
#include "cuda_utils.hpp"
#include "quantized.hpp"

namespace qwen {

class AttnLayer {
public:
    AttnLayer() = default;
    ~AttnLayer();

    void init(const ModelConfig& cfg, int layer_idx, int max_seq_len = 4096);
    void free_buffers();

    // Executes single-token GQA decode forward pass
    void forward(
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
    );

private:
    ModelConfig cfg_;
    int layer_idx_ = -1;
    int max_seq_len_ = 4096;

    // KV Cache tensors
    float* d_k_cache_ = nullptr; // max_seq_len x num_kv_heads x head_dim
    float* d_v_cache_ = nullptr; // max_seq_len x num_kv_heads x head_dim

    // Activation buffers
    float* d_q_ = nullptr;
    float* d_k_ = nullptr;
    float* d_v_ = nullptr;
    float* d_attn_out_ = nullptr;
    float* d_out_proj_ = nullptr;
};

} // namespace qwen

#endif // QWEN_LAYER_ATTN_HPP
