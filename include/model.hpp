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
    ~QwenModel();bool init(const ModelConfig&,const GgufLoader&);void free_buffers();int decode_step(int token_id,int pos,const CudaContext&);
private:
    struct Weights {const QuantTensor *norm=nullptr,*post_norm=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*o=nullptr,*qn=nullptr,*kn=nullptr;
      const QuantTensor *qkv=nullptr,*z=nullptr,*beta=nullptr,*alpha=nullptr,*conv=nullptr,*a=nullptr,*dt=nullptr,*gdn_norm=nullptr,*gdn_out=nullptr;
      const QuantTensor *router=nullptr,*eg=nullptr,*eu=nullptr,*ed=nullptr,*sgate=nullptr,*sg=nullptr,*su=nullptr,*sd=nullptr;};
    ModelConfig cfg_{};const GgufLoader*loader_=nullptr;const QuantTensor *embedding_=nullptr,*final_norm_=nullptr,*head_=nullptr;std::vector<Weights>w_;
    std::vector<std::unique_ptr<GdnLayer>>gdn_;std::vector<std::unique_ptr<AttnLayer>>attn_;std::vector<std::unique_ptr<MoeLayer>>moe_;
    float *x_=nullptr,*resid_=nullptr,*normed_=nullptr,*branch_=nullptr,*moe_out_=nullptr,*logits_=nullptr,*host_logits_=nullptr;int next_pos_=0;
};
}
#endif
