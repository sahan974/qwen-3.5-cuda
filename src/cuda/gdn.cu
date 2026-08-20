#include "gdn.hpp"
#include "cuda_utils.hpp"
#include <cmath>
namespace qwen {

__global__ void conv_step_kernel(const float* x,const float* w,float* hist,float* y,int c,int ks){
    int ch=blockIdx.x*blockDim.x+threadIdx.x;if(ch>=c)return;float sum=0;
    for(int j=0;j<ks-1;++j)sum+=hist[(size_t)ch*(ks-1)+j]*w[(size_t)ch*ks+j];
    sum+=x[ch]*w[(size_t)ch*ks+ks-1];
    for(int j=0;j<ks-2;++j)hist[(size_t)ch*(ks-1)+j]=hist[(size_t)ch*(ks-1)+j+1];
    hist[(size_t)ch*(ks-1)+ks-2]=x[ch];y[ch]=sum/(1.0f+expf(-sum));
}
void gdn_conv_step(const float*x,const float*w,float*h,float*y,int c,int ks,cudaStream_t s){
    conv_step_kernel<<<(c+255)/256,256,0,s>>>(x,w,h,y,c,ks);CUDA_CHECK(cudaGetLastError());
}

__global__ void qk_norm_kernel(float*q,float*k,int d,float eps){
    int h=blockIdx.x;float sq=0,sk=0;for(int i=threadIdx.x;i<d;i+=blockDim.x){float a=q[h*d+i],b=k[h*d+i];sq+=a*a;sk+=b*b;}
    for(int o=16;o;o>>=1){sq+=__shfl_down_sync(0xffffffff,sq,o);sk+=__shfl_down_sync(0xffffffff,sk,o);}
    __shared__ float pq[8],pk[8],rq,rk;if((threadIdx.x&31)==0){pq[threadIdx.x>>5]=sq;pk[threadIdx.x>>5]=sk;}__syncthreads();
    if(threadIdx.x<32){sq=threadIdx.x<blockDim.x/32?pq[threadIdx.x]:0;sk=threadIdx.x<blockDim.x/32?pk[threadIdx.x]:0;
      for(int o=16;o;o>>=1){sq+=__shfl_down_sync(0xffffffff,sq,o);sk+=__shfl_down_sync(0xffffffff,sk,o);}if(threadIdx.x==0){rq=rsqrtf(sq+eps);rk=rsqrtf(sk+eps);}}
    __syncthreads();float scale=rsqrtf((float)d);for(int i=threadIdx.x;i<d;i+=blockDim.x){q[h*d+i]*=rq*scale;k[h*d+i]*=rk;}
}
void gdn_qk_normalize(float*q,float*k,int h,int d,float e,cudaStream_t s){qk_norm_kernel<<<h,128,0,s>>>(q,k,d,e);CUDA_CHECK(cudaGetLastError());}

__global__ void gates_kernel(const float*b,const float*al,const float*a,const float*dt,float*bo,float*dec,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){bo[i]=1/(1+expf(-b[i]));float x=al[i]+dt[i];float sp=x>20?x:log1pf(expf(x));dec[i]=expf(a[i]*sp);}}
void gdn_gate_values(const float*b,const float*al,const float*a,const float*dt,float*bo,float*d,int n,cudaStream_t s){gates_kernel<<<(n+255)/256,256,0,s>>>(b,al,a,dt,bo,d,n);CUDA_CHECK(cudaGetLastError());}

__global__ void recurrent_kernel(float*S,const float*q,const float*k,const float*v,const float*beta,const float*decay,float*out,int kh,int vh,int dk,int dv){
    // Qwen3.5 GGUF tensors tile value heads as [subgroup][key-head].
    int h=blockIdx.x, j=threadIdx.x;if(j>=dv)return;int source=h%kh;const float*qh=q+source*dk;const float*khp=k+source*dk;float*Sh=S+(size_t)h*dk*dv;
    float mem=0;for(int i=0;i<dk;++i){float&cell=Sh[(size_t)i*dv+j];cell*=decay[h];mem+=khp[i]*cell;}
    float delta=(v[h*dv+j]-mem)*beta[h];for(int i=0;i<dk;++i)Sh[(size_t)i*dv+j]+=khp[i]*delta;
    float o=0;for(int i=0;i<dk;++i)o+=qh[i]*Sh[(size_t)i*dv+j];out[h*dv+j]=o;
}
void gdn_recurrent_step(float*S,const float*q,const float*k,const float*v,const float*b,const float*d,float*out,int kh,int vh,int dk,int dv,cudaStream_t s){
    if(dv>256)throw std::runtime_error("GDN value dimension exceeds kernel limit");recurrent_kernel<<<vh,256,0,s>>>(S,q,k,v,b,d,out,kh,vh,dk,dv);CUDA_CHECK(cudaGetLastError());
}

__global__ void norm_gate_kernel(const float*x,const float*z,const float*w,float*out,int d,float eps){int h=blockIdx.x;float ss=0;for(int i=threadIdx.x;i<d;i+=blockDim.x){float a=x[h*d+i];ss+=a*a;}for(int o=16;o;o>>=1)ss+=__shfl_down_sync(0xffffffff,ss,o);__shared__ float part[8],r;if((threadIdx.x&31)==0)part[threadIdx.x>>5]=ss;__syncthreads();if(threadIdx.x<32){ss=threadIdx.x<blockDim.x/32?part[threadIdx.x]:0;for(int o=16;o;o>>=1)ss+=__shfl_down_sync(0xffffffff,ss,o);if(threadIdx.x==0)r=rsqrtf(ss/d+eps);}__syncthreads();for(int i=threadIdx.x;i<d;i+=blockDim.x){float g=z[h*d+i];out[h*d+i]=x[h*d+i]*r*w[i]*(g/(1+expf(-g)));}}
void gdn_norm_gate(const float*x,const float*z,const float*w,float*out,int h,int d,float e,cudaStream_t s){norm_gate_kernel<<<h,128,0,s>>>(x,z,w,out,d,e);CUDA_CHECK(cudaGetLastError());}
}
