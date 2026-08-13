#ifndef QWEN_LAYER_MOE_HPP
#define QWEN_LAYER_MOE_HPP
#include "config.hpp"
#include "cuda_utils.hpp"
#include "quantized.hpp"
namespace qwen {
class MoeLayer {
public:
    ~MoeLayer();void init(const ModelConfig&,int);void free_buffers();
    void forward(const float*x,float*out,const QuantTensor&router,const QuantTensor&eg,const QuantTensor&eu,const QuantTensor&ed,
                 const QuantTensor&sgate,const QuantTensor&sg,const QuantTensor&su,const QuantTensor&sd,const CudaContext&ctx);
private:
    ModelConfig cfg_{};int layer_idx_=-1;float *router_=nullptr,*gate_=nullptr,*up_=nullptr,*hidden_=nullptr,*expert_out_=nullptr,*accum_=nullptr,*sgate_=nullptr,*shared_gate_=nullptr,*shared_up_=nullptr,*shared_hidden_=nullptr,*shared_out_=nullptr;
    int*top_idx_=nullptr;float*top_weight_=nullptr;int*h_idx_=nullptr;float*h_weight_=nullptr;
};
}
#endif
