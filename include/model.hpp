#ifndef QWEN_MODEL_HPP
#define QWEN_MODEL_HPP

#include "config.hpp"
#include "cuda_utils.hpp"
#include "loader_gguf.hpp"
#include "layer_gdn.hpp"
#include "layer_attn.hpp"
#include "layer_moe.hpp"

#include <memory>
#include <vector>

namespace qwen {

class QwenModel {
public:
    ~QwenModel();

    // --- Core Operations ---
    bool init(const ModelConfig& cfg, const GgufLoader& loader);
    void free_buffers();

    // --- Generation & Decoding ---
    const float* decode_logits(int token_id, int pos, const CudaContext& ctx);
    int          decode_step(int token_id, int pos, const CudaContext& ctx);

    // --- Accessors ---
    int vocab_size() const { return cfg_.vocab_size; }

private:
    // --- Layer Weights Structure ---
    struct Weights {
        // Normalization
        const QuantTensor* norm      = nullptr;
        const QuantTensor* post_norm = nullptr;

        // Attention (Full/Sliding)
        const QuantTensor* q         = nullptr;
        const QuantTensor* k         = nullptr;
        const QuantTensor* v         = nullptr;
        const QuantTensor* o         = nullptr;
        const QuantTensor* qn        = nullptr;
        const QuantTensor* kn        = nullptr;

        // Gated DeltaNet (GDN)
        const QuantTensor* qkv       = nullptr;
        const QuantTensor* z         = nullptr;
        const QuantTensor* beta      = nullptr;
        const QuantTensor* alpha     = nullptr;
        const QuantTensor* conv      = nullptr;
        const QuantTensor* a         = nullptr;
        const QuantTensor* dt        = nullptr;
        const QuantTensor* gdn_norm  = nullptr;
        const QuantTensor* gdn_out   = nullptr;

        // Mixture of Experts (MoE)
        const QuantTensor* router    = nullptr;
        const QuantTensor* eg        = nullptr;
        const QuantTensor* eu        = nullptr;
        const QuantTensor* ed        = nullptr;
        const QuantTensor* sgate     = nullptr;
        const QuantTensor* sg        = nullptr;
        const QuantTensor* su        = nullptr;
        const QuantTensor* sd        = nullptr;
    };

    // --- Model Configuration & Base Tensors ---
    ModelConfig         cfg_{};
    const GgufLoader*   loader_       = nullptr;

    const QuantTensor*  embedding_    = nullptr;
    const QuantTensor*  final_norm_   = nullptr;
    const QuantTensor*  head_         = nullptr;

    std::vector<Weights> w_;

    // --- Layer State Managers ---
    std::vector<std::unique_ptr<GdnLayer>>  gdn_;
    std::vector<std::unique_ptr<AttnLayer>> attn_;
    std::vector<std::unique_ptr<MoeLayer>>  moe_;

    // --- Inference Context Buffers ---
    float* x_           = nullptr;
    float* resid_       = nullptr;
    float* normed_      = nullptr;
    float* branch_      = nullptr;
    float* moe_out_     = nullptr;
    float* logits_      = nullptr;
    float* host_logits_ = nullptr;

    int next_pos_ = 0;
};

} // namespace qwen

#endif // QWEN_MODEL_HPP
