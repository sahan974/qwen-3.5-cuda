#ifndef QWEN_LAYER_GDN_HPP
#define QWEN_LAYER_GDN_HPP

#include "config.hpp"
#include "cuda_utils.hpp"
#include "quantized.hpp"

namespace qwen {

    class GdnLayer {
    public:
        ~GdnLayer();

        void init(const ModelConfig& cfg, int layer_idx);

        void free_buffers();

        void forward(
            const float*       x_norm,
            float*             mixer_out,
            const QuantTensor& qkv_w,
            const QuantTensor& z_w,
            const QuantTensor& beta_w,
            const QuantTensor& alpha_w,
            const QuantTensor& out_w,
            const float*       conv_w,
            const float*       a,
            const float*       dt_bias,
            const float*       norm_w,
            const CudaContext& ctx
        );

    private:
        ModelConfig cfg_{};
        int         layer_idx_ = -1;

        // --- State & History Buffers ---
        float* state_        = nullptr;
        float* conv_history_ = nullptr;

        // --- Intermediate Projection Buffers ---
        float* qkv_          = nullptr;
        float* z_            = nullptr;
        float* beta_raw_     = nullptr;
        float* alpha_raw_    = nullptr;

        // --- Processed State Buffers ---
        float* conv_out_     = nullptr;
        float* beta_         = nullptr;
        float* decay_        = nullptr;
        float* core_         = nullptr;
        float* normed_       = nullptr;
    };

} // namespace qwen

#endif // QWEN_LAYER_GDN_HPP
