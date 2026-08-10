#ifndef CONFIG_HPP
#ifndef QWEN_CONFIG_HPP
#define QWEN_CONFIG_HPP

#include <cstdint>

namespace qwen {

struct ModelConfig {
    int32_t d_model;             // Hidden size (2048)
    int32_t n_layers;            // Decoder layers count (40)
    int32_t vocab_size;          // Vocabulary size (248320)
    
    // Gated Grouped Query Attention (GQA) Parameters
    int32_t num_heads;           // Query heads (16)
    int32_t num_kv_heads;        // KV heads (2)
    int32_t head_dim;            // Head dimension (256)
    int32_t partial_rope_dim;    // RoPE rotation dimension (64)
    
    // Gated DeltaNet (Linear Attention) Parameters
    int32_t gdn_num_heads;       // Linear attention heads (32)
    int32_t gdn_key_dim;         // Key dimension (128)
    int32_t gdn_value_dim;       // Value dimension (128)
    int32_t conv_kernel_size;    // Causal depthwise conv kernel (4)
    
    // Mixture of Experts (MoE) Parameters
    int32_t num_experts;         // Total experts count (256)
    int32_t num_experts_per_tok; // Top-k routing (8)
    int32_t moe_intermediate_dim;// Expert intermediate size (512)
    int32_t shared_expert_dim;   // Shared expert intermediate size (2048)

    // Factory Configurations
    static ModelConfig real_config() {
        ModelConfig cfg;
        cfg.d_model = 2048;
        cfg.n_layers = 40;
        cfg.vocab_size = 248320;
        
        cfg.num_heads = 16;
        cfg.num_kv_heads = 2;
        cfg.head_dim = 256;
        cfg.partial_rope_dim = 64;
        
        cfg.gdn_num_heads = 32;
        cfg.gdn_key_dim = 128;
        cfg.gdn_value_dim = 128;
        cfg.conv_kernel_size = 4;
        
        cfg.num_experts = 256;
        cfg.num_experts_per_tok = 8;
        cfg.moe_intermediate_dim = 512;
        cfg.shared_expert_dim = 2048;
        return cfg;
    }

    static ModelConfig test_config() {
        ModelConfig cfg;
        cfg.d_model = 256;
        cfg.n_layers = 4;
        cfg.vocab_size = 1000;
        
        cfg.num_heads = 4;
        cfg.num_kv_heads = 1;
        cfg.head_dim = 64;
        cfg.partial_rope_dim = 16;
        
        cfg.gdn_num_heads = 4;
        cfg.gdn_key_dim = 32;
        cfg.gdn_value_dim = 32;
        cfg.conv_kernel_size = 4;
        
        cfg.num_experts = 8;
        cfg.num_experts_per_tok = 2;
        cfg.moe_intermediate_dim = 64;
        cfg.shared_expert_dim = 256;
        return cfg;
    }
};

// Derived Dimension Helpers
inline int32_t cfg_key_dim(const ModelConfig& cfg) {
    return cfg.gdn_num_heads * cfg.gdn_key_dim;
}

inline int32_t cfg_value_dim(const ModelConfig& cfg) {
    return cfg.gdn_num_heads * cfg.gdn_value_dim;
}

inline int32_t cfg_conv_dim(const ModelConfig& cfg) {
    return 2 * cfg_key_dim(cfg);
}

inline bool cfg_is_full_attn(int32_t layer_idx) {
    // Hybrid 3:1 pattern - every 4th layer (3, 7, 11, 15...) is GQA full attention
    return (layer_idx + 1) % 4 == 0;
}

} // namespace qwen

#endif // CONFIG_HPP
#endif // QWEN_CONFIG_HPP
