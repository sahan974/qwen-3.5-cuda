#include "gdn.hpp"
#include "matmul_q4.hpp"
#include "rope.hpp"
#include "cuda_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace qwen;
namespace {
void check_cuda(bool ok,const char*what){if(!ok)throw std::runtime_error(what);}
void half_one(std::vector<uint8_t>&b,size_t off){b[off]=0x00;b[off+1]=0x3c;}
float max_error(const std::vector<float>&a,const std::vector<float>&b){float e=0;for(size_t i=0;i<a.size();++i)e=std::max(e,std::fabs(a[i]-b[i]));return e;}
void test_quant(GgmlType type,int bytes,std::vector<uint8_t> raw,const std::vector<float>&expected){QuantTensor t;t.name="fixture";t.type=type;t.shape={256};t.num_elements=256;t.size_bytes=bytes;CUDA_CHECK(cudaMalloc(&t.device_ptr,bytes));CUDA_CHECK(cudaMemcpy(t.device_ptr,raw.data(),bytes,cudaMemcpyHostToDevice));float*d=nullptr;CUDA_CHECK(cudaMalloc(&d,256*sizeof(float)));dequantize_row(t,d,0,256);std::vector<float>got(256);CUDA_CHECK(cudaMemcpy(got.data(),d,256*sizeof(float),cudaMemcpyDeviceToHost));cudaFree(d);cudaFree(t.device_ptr);check_cuda(max_error(got,expected)<1e-6f,"quantized block layout mismatch");}
void quant_tests(){std::vector<float>expected(256);for(int i=0;i<256;++i)expected[i]=(i%64)<32?1.f:2.f;
 std::vector<uint8_t>q4(144);half_one(q4,0);for(int i=0;i<4;++i)q4[4+i]=1;for(int i=8;i<12;++i)q4[4+i]=1;std::fill(q4.begin()+16,q4.end(),0x21);test_quant(GgmlType::Q4_K,144,q4,expected);
 std::vector<uint8_t>q5(176);half_one(q5,0);for(int i=0;i<4;++i)q5[4+i]=1;for(int i=8;i<12;++i)q5[4+i]=1;std::fill(q5.begin()+48,q5.end(),0x21);test_quant(GgmlType::Q5_K,176,q5,expected);
 std::vector<uint8_t>q6(210);std::fill(q6.begin(),q6.begin()+128,0x11);std::fill(q6.begin()+128,q6.begin()+192,0xaa);std::fill(q6.begin()+192,q6.begin()+208,1);half_one(q6,208);test_quant(GgmlType::Q6_K,210,q6,std::vector<float>(256,1.f));}
void gdn_test(){constexpr int kh=1,vh=2,dk=4,dv=3;std::vector<float>S(vh*dk*dv),q={.2f,.3f,.4f,.5f},k={.1f,-.2f,.3f,.4f},v={1,2,3,-1,2,-3},beta={.25f,.75f},decay={.8f,.9f},ref(dv*vh);
 for(int h=0;h<vh;++h)for(int j=0;j<dv;++j){float mem=0;for(int i=0;i<dk;++i){auto&cell=S[(h*dk+i)*dv+j];cell*=decay[h];mem+=k[i]*cell;}float delta=(v[h*dv+j]-mem)*beta[h];for(int i=0;i<dk;++i)S[(h*dk+i)*dv+j]+=k[i]*delta;for(int i=0;i<dk;++i)ref[h*dv+j]+=q[i]*S[(h*dk+i)*dv+j];}
 float *ds,*dq,*dkp,*dv_,*db,*dd,*dout;CUDA_CHECK(cudaMalloc(&ds,S.size()*sizeof(float)));CUDA_CHECK(cudaMemset(ds,0,S.size()*sizeof(float)));CUDA_CHECK(cudaMalloc(&dq,q.size()*sizeof(float)));CUDA_CHECK(cudaMalloc(&dkp,k.size()*sizeof(float)));CUDA_CHECK(cudaMalloc(&dv_,v.size()*sizeof(float)));CUDA_CHECK(cudaMalloc(&db,beta.size()*sizeof(float)));CUDA_CHECK(cudaMalloc(&dd,decay.size()*sizeof(float)));CUDA_CHECK(cudaMalloc(&dout,ref.size()*sizeof(float)));CUDA_CHECK(cudaMemcpy(dq,q.data(),q.size()*sizeof(float),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dkp,k.data(),k.size()*sizeof(float),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dv_,v.data(),v.size()*sizeof(float),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(db,beta.data(),beta.size()*sizeof(float),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dd,decay.data(),decay.size()*sizeof(float),cudaMemcpyHostToDevice));gdn_recurrent_step(ds,dq,dkp,dv_,db,dd,dout,kh,vh,dk,dv,nullptr);std::vector<float>got(ref.size());CUDA_CHECK(cudaMemcpy(got.data(),dout,got.size()*sizeof(float),cudaMemcpyDeviceToHost));check_cuda(max_error(got,ref)<1e-6f,"GDN delta update mismatch");for(void*p:{ds,dq,dkp,dv_,db,dd,dout})cudaFree(p);}
}
int main(){try{quant_tests();gdn_test();std::cout<<"PASS: Q4_K/Q5_K/Q6_K layouts and GDN recurrence\n";return 0;}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
