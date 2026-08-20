#include "cuda_utils.hpp"
#include <cmath>
namespace qwen {
__device__ float warp_sum_attn(float v){for(int o=16;o;o>>=1)v+=__shfl_down_sync(0xffffffff,v,o);return v;}
__global__ void split_qg_kernel(const float*in,float*q,float*g,int h,int d){int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx<h*d){int head=idx/d,j=idx%d;q[idx]=in[head*2*d+j];g[idx]=in[head*2*d+d+j];}}
void attn_split_qg(const float*in,float*q,float*g,int h,int d,cudaStream_t s){int n=h*d;split_qg_kernel<<<(n+255)/256,256,0,s>>>(in,q,g,h,d);CUDA_CHECK(cudaGetLastError());}
__global__ void head_norm_kernel(float*x,const float*w,int d,float eps){int h=blockIdx.x;float ss=0;for(int i=threadIdx.x;i<d;i+=blockDim.x){float v=x[h*d+i];ss+=v*v;}ss=warp_sum_attn(ss);__shared__ float p[8],r;if((threadIdx.x&31)==0)p[threadIdx.x>>5]=ss;__syncthreads();if(threadIdx.x<32){ss=threadIdx.x<blockDim.x/32?p[threadIdx.x]:0;ss=warp_sum_attn(ss);if(threadIdx.x==0)r=rsqrtf(ss/d+eps);}__syncthreads();for(int i=threadIdx.x;i<d;i+=blockDim.x)x[h*d+i]*=r*w[i];}
void attn_head_norm(float*x,const float*w,int h,int d,float e,cudaStream_t s){head_norm_kernel<<<h,256,0,s>>>(x,w,d,e);CUDA_CHECK(cudaGetLastError());}
__global__ void score_kernel(const float*q,const float*k,float*scores,int H,int Hkv,int D,int pos){int h=blockIdx.x,t=blockIdx.y,kv=h/(H/Hkv);float sum=0;for(int i=threadIdx.x;i<D;i+=blockDim.x)sum+=q[h*D+i]*k[((size_t)t*Hkv+kv)*D+i];sum=warp_sum_attn(sum);if(threadIdx.x==0)scores[h*(pos+1)+t]=sum*rsqrtf((float)D);}
void attn_scores(const float*q,const float*k,float*s,int H,int Hkv,int D,int pos,cudaStream_t st){score_kernel<<<dim3(H,pos+1),32,0,st>>>(q,k,s,H,Hkv,D,pos);CUDA_CHECK(cudaGetLastError());}
__global__ void softmax_kernel(float*s,int n){float*r=s+blockIdx.x*n;float mx=-INFINITY;for(int i=threadIdx.x;i<n;i+=blockDim.x)mx=fmaxf(mx,r[i]);for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));__shared__ float pm[8],m,sum;if((threadIdx.x&31)==0)pm[threadIdx.x>>5]=mx;__syncthreads();if(threadIdx.x<32){mx=threadIdx.x<blockDim.x/32?pm[threadIdx.x]:-INFINITY;for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));if(threadIdx.x==0)m=mx;}__syncthreads();float ss=0;for(int i=threadIdx.x;i<n;i+=blockDim.x){float v=expf(r[i]-m);r[i]=v;ss+=v;}ss=warp_sum_attn(ss);if((threadIdx.x&31)==0)pm[threadIdx.x>>5]=ss;__syncthreads();if(threadIdx.x<32){ss=threadIdx.x<blockDim.x/32?pm[threadIdx.x]:0;ss=warp_sum_attn(ss);if(threadIdx.x==0)sum=ss;}__syncthreads();for(int i=threadIdx.x;i<n;i+=blockDim.x)r[i]/=sum;}
void attn_softmax(float*s,int H,int n,cudaStream_t st){softmax_kernel<<<H,256,0,st>>>(s,n);CUDA_CHECK(cudaGetLastError());}
__global__ void value_kernel(const float*s,const float*v,float*out,int H,int Hkv,int D,int pos){int h=blockIdx.x,j=threadIdx.x,kv=h/(H/Hkv);if(j<D){float z=0;for(int t=0;t<=pos;++t)z+=s[h*(pos+1)+t]*v[((size_t)t*Hkv+kv)*D+j];out[h*D+j]=z;}}
void attn_values(const float*s,const float*v,float*out,int H,int Hkv,int D,int pos,cudaStream_t st){value_kernel<<<H,256,0,st>>>(s,v,out,H,Hkv,D,pos);CUDA_CHECK(cudaGetLastError());}
__global__ void sigmoid_mul_kernel(const float*x,const float*g,float*out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=x[i]/(1+expf(-g[i]));}
void attn_sigmoid_gate(const float*x,const float*g,float*out,int n,cudaStream_t s){sigmoid_mul_kernel<<<(n+255)/256,256,0,s>>>(x,g,out,n);CUDA_CHECK(cudaGetLastError());}
}
