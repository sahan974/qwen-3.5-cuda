#ifndef QWEN_LAYER_GDN_HPP
#define QWEN_LAYER_GDN_HPP

#include "config.hpp"
#include "cuda_utils.hpp"
#include "quantized.hpp"

namespace qwen {

class GdnLayer {
public:
    GdnLayer() = default;
    ~GdnLayer();

    void init(const ModelConfig& cfg, int layer_idx);
    void free_buffers();

    // Executes single-token recurrent forward pass for a GDN layer
    void forward(
        const float* x_in,
        float* x_out,
        const QuantTensor& in_proj_w,
        const QuantTensor& out_proj_w,
        const float* norm_weight,
        const CudaContext& ctx
    );

private:
    ModelConfig cfg_;
    int layer_idx_ = -1;

    // Device workspace buffers
    float* d_state_ = nullptr;     // Recurrent state tensor S_t
    float* d_in_proj_ = nullptr;   // In-projection activations
    float* d_gdn_out_ = nullptr;   // GDN output activations
    float* d_norm_out_ = nullptr;  // Gated RMSNorm activations
    float* d_out_proj_ = nullptr;  // Out-projection output
};

} // namespace qwen

#endif // QWEN_LAYER_GDN_HPP
