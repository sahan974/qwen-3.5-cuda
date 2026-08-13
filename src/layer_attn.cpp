#include "layer_attn.hpp"
#include "matmul_q4.hpp"
#include "rope.hpp"
#include <stdexcept>
namespace qwen {
void attn_split_qg(const float*,float*,float*,int,int,cudaStream_t);void attn_head_norm(float*,const float*,int,int,float,cudaStream_t);
void attn_scores(const float*,const float*,float*,int,int,int,int,cudaStream_t);void attn_softmax(float*,int,int,cudaStream_t);
void attn_values(const float*,const float*,float*,int,int,int,int,cudaStream_t);void attn_sigmoid_gate(const float*,const float*,float*,int,cudaStream_t);
AttnLayer::~AttnLayer(){free_buffers();}
void AttnLayer::free_buffers(){for(float**p:{&k_cache_,&v_cache_,&qg_,&q_,&gate_,&k_,&v_,&scores_,&attn_,&gated_})if(*p){cudaFree(*p);*p=nullptr;}}
void AttnLayer::init(const ModelConfig&c,int li){free_buffers();cfg_=c;layer_idx_=li;int qd=c.num_heads*c.head_dim,kd=c.num_kv_heads*c.head_dim;
 CUDA_CHECK(cudaMalloc(&k_cache_,(size_t)c.max_seq_len*kd*sizeof(float)));CUDA_CHECK(cudaMalloc(&v_cache_,(size_t)c.max_seq_len*kd*sizeof(float)));
 CUDA_CHECK(cudaMalloc(&qg_,(size_t)2*qd*sizeof(float)));CUDA_CHECK(cudaMalloc(&q_,qd*sizeof(float)));CUDA_CHECK(cudaMalloc(&gate_,qd*sizeof(float)));CUDA_CHECK(cudaMalloc(&k_,kd*sizeof(float)));CUDA_CHECK(cudaMalloc(&v_,kd*sizeof(float)));
 CUDA_CHECK(cudaMalloc(&scores_,(size_t)c.num_heads*c.max_seq_len*sizeof(float)));CUDA_CHECK(cudaMalloc(&attn_,qd*sizeof(float)));CUDA_CHECK(cudaMalloc(&gated_,qd*sizeof(float)));
}
void AttnLayer::forward(const float*x,float*out,int pos,const QuantTensor&qw,const QuantTensor&kw,const QuantTensor&vw,const QuantTensor&ow,const float*qn,const float*kn,const CudaContext&ctx){
 if(pos<0||pos>=cfg_.max_seq_len)throw std::runtime_error("attention position exceeds configured context");int qd=cfg_.num_heads*cfg_.head_dim,kd=cfg_.num_kv_heads*cfg_.head_dim;
 matmul_dispatch(qw,x,qg_,2*qd,cfg_.d_model,ctx.stream());matmul_dispatch(kw,x,k_,kd,cfg_.d_model,ctx.stream());matmul_dispatch(vw,x,v_,kd,cfg_.d_model,ctx.stream());
 attn_split_qg(qg_,q_,gate_,cfg_.num_heads,cfg_.head_dim,ctx.stream());attn_head_norm(q_,qn,cfg_.num_heads,cfg_.head_dim,cfg_.rms_eps,ctx.stream());attn_head_norm(k_,kn,cfg_.num_kv_heads,cfg_.head_dim,cfg_.rms_eps,ctx.stream());
 rope_forward(q_,pos,cfg_.num_heads,cfg_.head_dim,cfg_.partial_rope_dim,cfg_.rope_theta,ctx.stream());rope_forward(k_,pos,cfg_.num_kv_heads,cfg_.head_dim,cfg_.partial_rope_dim,cfg_.rope_theta,ctx.stream());
 CUDA_CHECK(cudaMemcpyAsync(k_cache_+(size_t)pos*kd,k_,kd*sizeof(float),cudaMemcpyDeviceToDevice,ctx.stream()));CUDA_CHECK(cudaMemcpyAsync(v_cache_+(size_t)pos*kd,v_,kd*sizeof(float),cudaMemcpyDeviceToDevice,ctx.stream()));
 attn_scores(q_,k_cache_,scores_,cfg_.num_heads,cfg_.num_kv_heads,cfg_.head_dim,pos,ctx.stream());attn_softmax(scores_,cfg_.num_heads,pos+1,ctx.stream());attn_values(scores_,v_cache_,attn_,cfg_.num_heads,cfg_.num_kv_heads,cfg_.head_dim,pos,ctx.stream());
 attn_sigmoid_gate(attn_,gate_,gated_,qd,ctx.stream());matmul_dispatch(ow,gated_,out,cfg_.d_model,qd,ctx.stream());
}
}
