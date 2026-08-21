#ifndef QWEN_LAYER_MOE_HPP
#define QWEN_LAYER_MOE_HPP

#include "config.hpp"
#include "cuda_utils.hpp"
#include "quantized.hpp"

namespace qwen {

    class MoeLayer {
    public:
        ~MoeLayer();

        void init(const ModelConfig& cfg, int layer_idx);

        void free_buffers();

        void forward(
            const float*       x,
            float*             out,
            const QuantTensor& router,
            const QuantTensor& eg,
            const QuantTensor& eu,
            const QuantTensor& ed,
            const QuantTensor& sgate,
            const QuantTensor& sg,
            const QuantTensor& su,
            const QuantTensor& sd,
            const CudaContext& ctx
        );

    private:
        ModelConfig cfg_{};
        int         layer_idx_ = -1;

        // --- MoE Routing & Top-K Indices (Device & Host) ---
        float* router_   = nullptr;
        int*   top_idx_  = nullptr;
        float* top_weight_= nullptr;
        int*   h_idx_    = nullptr;
        float* h_weight_ = nullptr;

        // --- Sparse Expert Buffers ---
        float* gate_       = nullptr;
        float* up_         = nullptr;
        float* hidden_     = nullptr;
        float* expert_out_ = nullptr;
        float* accum_      = nullptr;

        // --- Shared Expert Buffers ---
        float* sgate_         = nullptr;
        float* shared_gate_   = nullptr;
        float* shared_up_     = nullptr;
        float* shared_hidden_ = nullptr;
        float* shared_out_    = nullptr;
    };

} // namespace qwen

#endif // QWEN_LAYER_MOE_HPP
