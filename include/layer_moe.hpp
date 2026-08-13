#ifndef QWEN_LAYER_MOE_HPP
#define QWEN_LAYER_MOE_HPP

#include "config.hpp"
#include "cuda_utils.hpp"
#include "quantized.hpp"

namespace qwen {

class MoeLayer {
public:
    MoeLayer() = default;
    ~MoeLayer();

    void init(const ModelConfig& cfg, int layer_idx);
    void free_buffers();

    // Executes single-token MoE layer forward pass
    void forward(
        const float* x_in,
        float* x_out,
        const QuantTensor& router_w,
        const QuantTensor& shared_expert_w,
        const CudaContext& ctx
    );

private:
    ModelConfig cfg_;
    int layer_idx_ = -1;

    // Device buffers
    float* d_router_logits_ = nullptr; // 256
    int* d_topk_indices_ = nullptr;   // 8
    float* d_topk_weights_ = nullptr;   // 8
    float* d_expert_outputs_ = nullptr; // 8 x d_model
    float* d_moe_out_ = nullptr;        // d_model
    float* d_shared_out_ = nullptr;     // d_model
};

} // namespace qwen

#endif // QWEN_LAYER_MOE_HPP
