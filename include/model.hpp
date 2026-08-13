#ifndef QWEN_MODEL_HPP
#define QWEN_MODEL_HPP

#include "config.hpp"
#include "cuda_utils.hpp"
#include "loader_gguf.hpp"
#include "layer_gdn.hpp"
#include "layer_attn.hpp"
#include "layer_moe.hpp"
#include <vector>
#include <memory>

namespace qwen {

class QwenModel {
public:
    QwenModel() = default;
    ~QwenModel();

    bool init(const ModelConfig& cfg, const GgufLoader& loader);
    void free_buffers();

    // Runs single-token autoregressive step and returns greedy next token ID
    int decode_step(int token_id, int pos, const CudaContext& ctx);

private:
    ModelConfig cfg_;
    
    std::vector<std::unique_ptr<GdnLayer>> gdn_layers_;
    std::vector<std::unique_ptr<AttnLayer>> attn_layers_;
    std::vector<std::unique_ptr<MoeLayer>> moe_layers_;

    // Device workspace buffers
    float* d_x_ = nullptr;        // Hidden activation vector (d_model)
    float* d_x_next_ = nullptr;   // Next hidden activation vector
    float* d_logits_ = nullptr;   // Final output logits (vocab_size)
    float* h_logits_ = nullptr;   // Host pinned buffer for sampling
};

} // namespace qwen

#endif // QWEN_MODEL_HPP
