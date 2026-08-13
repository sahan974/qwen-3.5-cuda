#ifndef QWEN_LAYER_ATTN_HPP
#define QWEN_LAYER_ATTN_HPP
#include "config.hpp"
#include "cuda_utils.hpp"
#include "quantized.hpp"
namespace qwen {
class AttnLayer {
public:
    ~AttnLayer();
    void init(const ModelConfig& cfg,int layer_idx);
    void free_buffers();
    void forward(const float*x_norm,float*mixer_out,int pos,const QuantTensor&q_w,const QuantTensor&k_w,
                 const QuantTensor&v_w,const QuantTensor&o_w,const float*q_norm,const float*k_norm,const CudaContext&ctx);
private:
    ModelConfig cfg_{};int layer_idx_=-1;
    float *k_cache_=nullptr,*v_cache_=nullptr,*qg_=nullptr,*q_=nullptr,*gate_=nullptr,*k_=nullptr,*v_=nullptr,*scores_=nullptr,*attn_=nullptr,*gated_=nullptr;
};
}
#endif
