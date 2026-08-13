#include "rope.hpp"
#include "cuda_utils.hpp"
#include <cmath>
namespace qwen {
__global__ void rope_kernel(float*x,int pos,int heads,int hd,int rd,float base){int h=blockIdx.x,i=threadIdx.x,half=rd/2;if(h>=heads||i>=half)return;float* p=x+h*hd;float theta=pos/powf(base,(2.0f*i)/rd),c=cosf(theta),s=sinf(theta),a=p[i],b=p[i+half];p[i]=a*c-b*s;p[i+half]=b*c+a*s;}
void rope_forward(float*x,int pos,int heads,int hd,int rd,float base,cudaStream_t s){rope_kernel<<<heads,rd/2,0,s>>>(x,pos,heads,hd,rd,base);CUDA_CHECK(cudaGetLastError());}
}
