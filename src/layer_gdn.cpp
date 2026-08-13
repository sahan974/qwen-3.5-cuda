#include "layer_gdn.hpp"
#include "gdn.hpp"
#include "matmul_q4.hpp"
namespace qwen {
GdnLayer::~GdnLayer(){free_buffers();}
void GdnLayer::free_buffers(){for(float**p:{&state_,&conv_history_,&qkv_,&z_,&beta_raw_,&alpha_raw_,&conv_out_,&beta_,&decay_,&core_,&normed_})if(*p){cudaFree(*p);*p=nullptr;}}
void GdnLayer::init(const ModelConfig&c,int li){free_buffers();cfg_=c;layer_idx_=li;int cd=cfg_conv_dim(c),vd=cfg_value_dim(c),vh=c.gdn_num_v_heads;
 CUDA_CHECK(cudaMalloc(&state_,(size_t)vh*c.gdn_key_dim*c.gdn_value_dim*sizeof(float)));CUDA_CHECK(cudaMemset(state_,0,(size_t)vh*c.gdn_key_dim*c.gdn_value_dim*sizeof(float)));
 CUDA_CHECK(cudaMalloc(&conv_history_,(size_t)cd*(c.conv_kernel_size-1)*sizeof(float)));CUDA_CHECK(cudaMemset(conv_history_,0,(size_t)cd*(c.conv_kernel_size-1)*sizeof(float)));
 CUDA_CHECK(cudaMalloc(&qkv_,cd*sizeof(float)));CUDA_CHECK(cudaMalloc(&z_,vd*sizeof(float)));CUDA_CHECK(cudaMalloc(&beta_raw_,vh*sizeof(float)));CUDA_CHECK(cudaMalloc(&alpha_raw_,vh*sizeof(float)));
 CUDA_CHECK(cudaMalloc(&conv_out_,cd*sizeof(float)));CUDA_CHECK(cudaMalloc(&beta_,vh*sizeof(float)));CUDA_CHECK(cudaMalloc(&decay_,vh*sizeof(float)));CUDA_CHECK(cudaMalloc(&core_,vd*sizeof(float)));CUDA_CHECK(cudaMalloc(&normed_,vd*sizeof(float)));
}
void GdnLayer::forward(const float*x,float*out,const QuantTensor&qkvw,const QuantTensor&zw,const QuantTensor&bw,const QuantTensor&aw,const QuantTensor&ow,const float*cw,const float*a,const float*dt,const float*nw,const CudaContext&ctx){
 int kd=cfg_key_dim(cfg_),vd=cfg_value_dim(cfg_),cd=cfg_conv_dim(cfg_),vh=cfg_.gdn_num_v_heads;
 matmul_dispatch(qkvw,x,qkv_,cd,cfg_.d_model,ctx.stream());matmul_dispatch(zw,x,z_,vd,cfg_.d_model,ctx.stream());matmul_dispatch(bw,x,beta_raw_,vh,cfg_.d_model,ctx.stream());matmul_dispatch(aw,x,alpha_raw_,vh,cfg_.d_model,ctx.stream());
 gdn_conv_step(qkv_,cw,conv_history_,conv_out_,cd,cfg_.conv_kernel_size,ctx.stream());float*q=conv_out_,*k=conv_out_+kd,*v=conv_out_+2*kd;
 gdn_qk_normalize(q,k,cfg_.gdn_num_k_heads,cfg_.gdn_key_dim,cfg_.rms_eps,ctx.stream());gdn_gate_values(beta_raw_,alpha_raw_,a,dt,beta_,decay_,vh,ctx.stream());
 gdn_recurrent_step(state_,q,k,v,beta_,decay_,core_,cfg_.gdn_num_k_heads,vh,cfg_.gdn_key_dim,cfg_.gdn_value_dim,ctx.stream());gdn_norm_gate(core_,z_,nw,normed_,vh,cfg_.gdn_value_dim,cfg_.rms_eps,ctx.stream());
 matmul_dispatch(ow,normed_,out,cfg_.d_model,vd,ctx.stream());
}
}
