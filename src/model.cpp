#include "model.hpp"
#include "ops.hpp"
#include "matmul_q4.hpp"
#include <iostream>
#include <algorithm>

namespace qwen {

QwenModel::~QwenModel() {
    free_buffers();
}

void QwenModel::free_buffers() {
    if (d_x_) { cudaFree(d_x_); d_x_ = nullptr; }
    if (d_x_next_) { cudaFree(d_x_next_); d_x_next_ = nullptr; }
    if (d_logits_) { cudaFree(d_logits_); d_logits_ = nullptr; }
    if (h_logits_) { cudaFreeHost(h_logits_); h_logits_ = nullptr; }

    gdn_layers_.clear();
    attn_layers_.clear();
    moe_layers_.clear();
}

bool QwenModel::init(const ModelConfig& cfg, const GgufLoader& loader) {
    cfg_ = cfg;
    free_buffers();

    CUDA_CHECK(cudaMalloc(&d_x_, cfg_.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_next_, cfg_.d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logits_, cfg_.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaHostAlloc(&h_logits_, cfg_.vocab_size * sizeof(float), cudaHostAllocDefault));

    // Initialize 40 decoder layers
    for (int i = 0; i < cfg_.n_layers; ++i) {
        if (cfg_is_full_attn(i)) {
            auto attn = std::make_unique<AttnLayer>();
            attn->init(cfg_, i);
            attn_layers_.push_back(std::move(attn));
        } else {
            auto gdn = std::make_unique<GdnLayer>();
            gdn->init(cfg_, i);
            gdn_layers_.push_back(std::move(gdn));
        }

        auto moe = std::make_unique<MoeLayer>();
        moe->init(cfg_, i);
        moe_layers_.push_back(std::move(moe));
    }

    std::cout << "QwenModel initialized with " << cfg_.n_layers << " layers successfully." << std::endl;
    return true;
}

int QwenModel::decode_step(int token_id, int pos, const CudaContext& ctx) {
    // 1. Embedding lookup (dummy zeros for demo)
    CUDA_CHECK(cudaMemsetAsync(d_x_, 0, cfg_.d_model * sizeof(float), ctx.stream()));

    // Dummy empty QuantTensor for testing layer connections
    QuantTensor dummy_w;
    dummy_w.device_ptr = d_x_;

    int gdn_idx = 0;
    int attn_idx = 0;

    // 2. Pass through 40 decoder layers
    for (int i = 0; i < cfg_.n_layers; ++i) {
        if (cfg_is_full_attn(i)) {
            attn_layers_[attn_idx]->forward(d_x_, d_x_next_, pos, dummy_w, dummy_w, dummy_w, dummy_w, d_x_, d_x_, ctx);
            attn_idx++;
        } else {
            gdn_layers_[gdn_idx]->forward(d_x_, d_x_next_, dummy_w, dummy_w, d_x_, ctx);
            gdn_idx++;
        }

        // Swap activation pointers
        std::swap(d_x_, d_x_next_);

        // MoE block
        moe_layers_[i]->forward(d_x_, d_x_next_, dummy_w, dummy_w, ctx);
        std::swap(d_x_, d_x_next_);
    }

    // 3. Final RMSNorm
    rmsnorm_forward(d_x_, d_x_, d_x_next_, cfg_.d_model, 1e-6f, ctx.stream());

    // 4. LM Head Logits GEMM
    matmul_w4a16(
        static_cast<const uint8_t*>(dummy_w.device_ptr),
        d_x_,
        d_x_,
        d_x_next_,
        d_logits_,
        cfg_.vocab_size,
        cfg_.d_model,
        32,
        ctx.stream()
    );

    // 5. Copy logits to host pinned memory for greedy sampling
    CUDA_CHECK(cudaMemcpyAsync(h_logits_, d_logits_, cfg_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    ctx.synchronize();

    // Argmax selection
    int next_token = 0;
    float max_logit = h_logits_[0];
    for (int v = 1; v < cfg_.vocab_size; ++v) {
        if (h_logits_[v] > max_logit) {
            max_logit = h_logits_[v];
            next_token = v;
        }
    }

    return next_token;
}

} // namespace qwen
