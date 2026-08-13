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
    void forward(const float* x_norm, float* mixer_out,
                 const QuantTensor& qkv_w, const QuantTensor& z_w,
                 const QuantTensor& beta_w, const QuantTensor& alpha_w,
                 const QuantTensor& out_w, const float* conv_w, const float* a,
                 const float* dt_bias, const float* norm_w, const CudaContext& ctx);
private:
    ModelConfig cfg_{}; int layer_idx_ = -1;
    float *state_=nullptr,*conv_history_=nullptr,*qkv_=nullptr,*z_=nullptr,*beta_raw_=nullptr,*alpha_raw_=nullptr;
    float *conv_out_=nullptr,*beta_=nullptr,*decay_=nullptr,*core_=nullptr,*normed_=nullptr;
};
}
#endif
