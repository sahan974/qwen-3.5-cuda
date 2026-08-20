#include "sampler.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace qwen;
int main(){try{
    const float logits[]={1.f,2.f,3.f};
    SamplingConfig greedy;Sampler a(greedy);if(a.sample(logits,3,{})!=2)throw std::runtime_error("greedy sampling failed");
    const float repeat_logits[]={3.f,2.f};SamplingConfig repeat;repeat.repetition_penalty=2.f;Sampler b(repeat);if(b.sample(repeat_logits,2,{0})!=1)throw std::runtime_error("repetition penalty failed");
    SamplingConfig stochastic;stochastic.temperature=.8f;stochastic.top_k=2;stochastic.top_p=.9f;stochastic.seed=123;Sampler c(stochastic),d(stochastic);std::vector<int> history;for(int i=0;i<32;++i){int x=c.sample(logits,3,history),y=d.sample(logits,3,history);if(x!=y||x==0)throw std::runtime_error("seeded top-k/top-p sampling failed");history.push_back(x);}
    std::cout<<"PASS: greedy, repetition penalty, top-k/top-p, and seeded sampling\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
