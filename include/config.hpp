#ifndef QWEN_CONFIG_HPP
#define QWEN_CONFIG_HPP

#include <cstdint>
#include <cmath>
#include <stdexcept>

namespace qwen {

struct ModelConfig {
    // --- Core Model Dimensions ---
    int32_t d_model     = 0;
    int32_t n_layers    = 0;
    int32_t vocab_size  = 0;
    int32_t max_seq_len = 4096;
    float   rms_eps     = 1e-6f;

    // --- Full Attention & RoPE ---
    int32_t num_heads        = 0;
    int32_t num_kv_heads     = 0;
    int32_t head_dim         = 0;
    int32_t partial_rope_dim = 0;
    float   rope_theta       = 10000000.0f;

    // --- Gated DeltaNet (GDN) ---
    int32_t gdn_num_k_heads  = 0;
    int32_t gdn_num_v_heads  = 0;
    int32_t gdn_key_dim      = 0;
    int32_t gdn_value_dim    = 0;
    int32_t conv_kernel_size = 0;

    // --- Mixture of Experts (MoE) ---
    int32_t num_experts          = 0;
    int32_t num_experts_per_tok  = 0;
    int32_t moe_intermediate_dim = 0;
    int32_t shared_expert_dim    = 0;
    int32_t full_attn_interval   = 4;

    static ModelConfig real_config() {
        ModelConfig c;

        c.d_model              = 2048;
        c.n_layers             = 40;
        c.vocab_size           = 248320;

        c.num_heads            = 16;
        c.num_kv_heads         = 2;
        c.head_dim             = 256;
        c.partial_rope_dim     = 64;

        c.gdn_num_k_heads      = 16;
        c.gdn_num_v_heads      = 32;
        c.gdn_key_dim          = 128;
        c.gdn_value_dim        = 128;
        c.conv_kernel_size     = 4;

        c.num_experts          = 256;
        c.num_experts_per_tok  = 8;
        c.moe_intermediate_dim = 512;
        c.shared_expert_dim    = 2048;

        return c;
    }

    void validate() const {
        // Core dimensions
        if (d_model <= 0 || n_layers <= 0 || vocab_size <= 0 || max_seq_len <= 0) {
            throw std::runtime_error("invalid core model dimensions");
        }

        // Normalization & RoPE
        if (!(rms_eps > 0.0f) || !std::isfinite(rms_eps)) {
            throw std::runtime_error("invalid normalization parameters");
        }
        if (!(rope_theta > 0.0f) || !std::isfinite(rope_theta)) {
            throw std::runtime_error("invalid RoPE parameters");
        }

        // Full Attention
        if (full_attn_interval <= 0) {
            throw std::runtime_error("invalid full-attention interval");
        }
        if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || head_dim > 256) {
            throw std::runtime_error("invalid full-attention head configuration");
        }
        if (num_heads % num_kv_heads != 0) {
            throw std::runtime_error("invalid full-attention head configuration (num_heads not divisible by num_kv_heads)");
        }
        if (partial_rope_dim <= 0 || partial_rope_dim > head_dim || partial_rope_dim % 2 != 0) {
            throw std::runtime_error("invalid rotary dimension");
        }

        // Gated DeltaNet (GDN)
        if (gdn_num_k_heads <= 0 || gdn_num_v_heads <= 0) {
            throw std::runtime_error("invalid GDN head grouping (must be > 0)");
        }
        if (gdn_num_v_heads % gdn_num_k_heads != 0) {
            throw std::runtime_error("invalid GDN head grouping (v_heads must be divisible by k_heads)");
        }
        if (gdn_key_dim <= 0 || gdn_value_dim <= 0 || gdn_value_dim > 256 || conv_kernel_size < 2) {
            throw std::runtime_error("invalid GDN dimensions");
        }

        // Mixture of Experts (MoE)
        if (moe_intermediate_dim <= 0 || shared_expert_dim <= 0) {
            throw std::runtime_error("invalid MoE dimensions");
        }
        if (num_experts <= 0 || num_experts_per_tok <= 0 || num_experts_per_tok > num_experts) {
            throw std::runtime_error("invalid MoE routing configuration");
        }
    }
};

// --- Helper Functions ---
inline int32_t cfg_key_dim(const ModelConfig& c) {
    return c.gdn_num_k_heads * c.gdn_key_dim;
}

inline int32_t cfg_value_dim(const ModelConfig& c) {
    return c.gdn_num_v_heads * c.gdn_value_dim;
}

inline int32_t cfg_conv_dim(const ModelConfig& c) {
    return (2 * cfg_key_dim(c)) + cfg_value_dim(c);
}

inline bool cfg_is_full_attn(const ModelConfig& c, int32_t layer) {
    return ((layer + 1) % c.full_attn_interval) == 0;
}

} // namespace qwen

#endif
