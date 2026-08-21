#include "layer_attn.hpp"
#include "matmul_q4.hpp"
#include "rope.hpp"

#include <stdexcept>

namespace qwen {

// --- External CUDA Kernel Declarations ---
void attn_split_qg(const float* qg, float* q, float* gate, int heads, int dim, cudaStream_t stream);
void attn_head_norm(float* x, const float* weight, int heads, int dim, float eps, cudaStream_t stream);
void attn_scores(const float* q, const float* k_cache, float* scores, int q_heads, int kv_heads, int dim, int pos, cudaStream_t stream);
void attn_softmax(float* scores, int heads, int valid_len, cudaStream_t stream);
void attn_values(const float* scores, const float* v_cache, float* out, int q_heads, int kv_heads, int dim, int pos, cudaStream_t stream);
void attn_sigmoid_gate(const float* attn, const float* gate, float* out, int dim, cudaStream_t stream);


// --- Lifecycle ---
AttnLayer::~AttnLayer() {
    free_buffers();
}

void AttnLayer::free_buffers() {
    float** buffers[] = {
        &k_cache_, &v_cache_, &qg_, &q_, &gate_,
        &k_, &v_, &scores_, &attn_, &gated_
    };

    for (float** p : buffers) {
        if (*p) {
            cudaFree(*p);
            *p = nullptr;
        }
    }
}

void AttnLayer::init(const ModelConfig& c, int li) {
    free_buffers();
    cfg_ = c;
    layer_idx_ = li;

    int qd = c.num_heads * c.head_dim;
    int kd = c.num_kv_heads * c.head_dim;

    // Allocate KV Caches
    CUDA_CHECK(cudaMalloc(&k_cache_, (size_t)c.max_seq_len * kd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&v_cache_, (size_t)c.max_seq_len * kd * sizeof(float)));

    // Allocate Projection Buffers
    CUDA_CHECK(cudaMalloc(&qg_,   (size_t)2 * qd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q_,    qd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gate_, qd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&k_,    kd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&v_,    kd * sizeof(float)));

    // Allocate Attention Computation Buffers
    CUDA_CHECK(cudaMalloc(&scores_, (size_t)c.num_heads * c.max_seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&attn_,   qd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gated_,  qd * sizeof(float)));
}


// --- Forward Pass ---
void AttnLayer::forward(
    const float*       x,
    float*             out,
    int                pos,
    const QuantTensor& qw,
    const QuantTensor& kw,
    const QuantTensor& vw,
    const QuantTensor& ow,
    const float*       qn,
    const float*       kn,
    const CudaContext& ctx
) {
    if (pos < 0 || pos >= cfg_.max_seq_len) {
        throw std::runtime_error("attention position exceeds configured context");
    }

    int qd = cfg_.num_heads * cfg_.head_dim;
    int kd = cfg_.num_kv_heads * cfg_.head_dim;
    cudaStream_t stream = ctx.stream();

    // 1. QKV Projections
    matmul_dispatch(qw, x, qg_, 2 * qd, cfg_.d_model, stream);
    matmul_dispatch(kw, x, k_,  kd,     cfg_.d_model, stream);
    matmul_dispatch(vw, x, v_,  kd,     cfg_.d_model, stream);

    // 2. Split Q and Gate, then apply Head Normalization
    attn_split_qg(qg_, q_, gate_, cfg_.num_heads, cfg_.head_dim, stream);
    attn_head_norm(q_, qn, cfg_.num_heads, cfg_.head_dim, cfg_.rms_eps, stream);
    attn_head_norm(k_, kn, cfg_.num_kv_heads, cfg_.head_dim, cfg_.rms_eps, stream);

    // 3. Apply Rotary Positional Embeddings (RoPE)
    rope_forward(q_, pos, cfg_.num_heads, cfg_.head_dim, cfg_.partial_rope_dim, cfg_.rope_theta, stream);
    rope_forward(k_, pos, cfg_.num_kv_heads, cfg_.head_dim, cfg_.partial_rope_dim, cfg_.rope_theta, stream);

    // 4. Update KV Cache
    CUDA_CHECK(cudaMemcpyAsync(k_cache_ + (size_t)pos * kd, k_, kd * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(v_cache_ + (size_t)pos * kd, v_, kd * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // 5. Scaled Dot-Product Attention
    attn_scores(q_, k_cache_, scores_, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, pos, stream);
    attn_softmax(scores_, cfg_.num_heads, pos + 1, stream);
    attn_values(scores_, v_cache_, attn_, cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim, pos, stream);

    // 6. Output Gating & Final Projection
    attn_sigmoid_gate(attn_, gate_, gated_, qd, stream);
    matmul_dispatch(ow, gated_, out, cfg_.d_model, qd, stream);
}

} // namespace qwen
