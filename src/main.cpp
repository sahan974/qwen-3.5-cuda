#include "config.hpp"
#include "cuda_utils.hpp"
#include "loader_gguf.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include "tokenizer.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
using qwen::QuantTensor;
namespace {
const QuantTensor& tensor(const qwen::GgufLoader&l,const std::string&n){auto*t=l.get_tensor(n);if(!t)throw std::runtime_error("cannot derive config; missing "+n);return*t;}
bool has(const qwen::GgufLoader& l, const std::string& name) { return l.get_tensor(name) != nullptr; }
bool has_any(const qwen::GgufLoader& l, const std::string& a, const std::string& b) {
    return has(l, a) || has(l, b);
}
int derive_base_layer_count(const qwen::GgufLoader& l, int full_attn_interval) {
    if (full_attn_interval <= 0) throw std::runtime_error("invalid full-attention interval");
    int count = 0;
    for (;; ++count) {
        const std::string p = "blk." + std::to_string(count) + ".";
        const bool common = has(l, p + "attn_norm.weight") &&
            has_any(l, p + "post_attention_norm.weight", p + "attn_post_norm.weight") &&
            has(l, p + "ffn_gate_inp.weight") && has(l, p + "ffn_gate_exps.weight") &&
            has(l, p + "ffn_up_exps.weight") && has(l, p + "ffn_down_exps.weight") &&
            has(l, p + "ffn_gate_inp_shexp.weight") && has(l, p + "ffn_gate_shexp.weight") &&
            has(l, p + "ffn_up_shexp.weight") && has(l, p + "ffn_down_shexp.weight");
        const bool full = (count + 1) % full_attn_interval == 0;
        const bool mixer = full
            ? has(l, p + "attn_q.weight") && has(l, p + "attn_k.weight") &&
              has(l, p + "attn_v.weight") && has(l, p + "attn_output.weight") &&
              has(l, p + "attn_q_norm.weight") && has(l, p + "attn_k_norm.weight")
            : has(l, p + "attn_qkv.weight") && has(l, p + "attn_gate.weight") &&
              has(l, p + "ssm_beta.weight") && has(l, p + "ssm_alpha.weight") &&
              has(l, p + "ssm_conv1d.weight") && has(l, p + "ssm_a") &&
              has(l, p + "ssm_dt.bias") && has(l, p + "ssm_norm.weight") &&
              has(l, p + "ssm_out.weight");
        if (!common || !mixer) break;
    }
    if (count == 0) throw std::runtime_error("could not find a complete base-model layer");
    return count;
}
qwen::ModelConfig config_from(const qwen::GgufLoader& l, int ctx) {
    const std::string architecture = l.get_meta_string("general.architecture");
    if (architecture != "qwen35moe")
        throw std::runtime_error("this runtime requires a qwen35moe GGUF, got '" + architecture + "'");

    qwen::ModelConfig c = qwen::ModelConfig::real_config();
    const auto& emb = tensor(l, "token_embd.weight");
    if (emb.shape.size() != 2) throw std::runtime_error("token embedding must be 2-D");
    c.d_model = static_cast<int>(emb.shape[0]);
    c.vocab_size = static_cast<int>(emb.shape[1]);
    c.max_seq_len = ctx;
    c.rms_eps = static_cast<float>(l.get_meta_float("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6));
    c.full_attn_interval = l.get_meta_int("qwen35moe.full_attention_interval", 4);
    c.n_layers = derive_base_layer_count(l, c.full_attn_interval);
    const int reported_layers = l.get_meta_int("qwen35moe.block_count", c.n_layers);
    if (reported_layers != c.n_layers) {
        std::cerr << "Ignoring " << (reported_layers - c.n_layers)
                  << " optional/non-base block(s); using " << c.n_layers << " complete model layers.\n";
    }
    c.rope_theta = static_cast<float>(l.get_meta_float("qwen35moe.rope.freq_base", 10000000.0));

    const auto& qkv = tensor(l, "blk.0.attn_qkv.weight");
    const auto& z = tensor(l, "blk.0.attn_gate.weight");
    const auto& conv = tensor(l, "blk.0.ssm_conv1d.weight");
    const auto& alpha = tensor(l, "blk.0.ssm_alpha.weight");
    const auto& gn = tensor(l, "blk.0.ssm_norm.weight");
    if (qkv.shape.size()!=2 || z.shape.size()!=2 || conv.shape.size()!=2 || alpha.shape.size()!=2 || gn.shape.size()!=1 ||
        qkv.shape[0]!=c.d_model || z.shape[0]!=c.d_model || alpha.shape[0]!=c.d_model || conv.shape[1]!=qkv.shape[1])
        throw std::runtime_error("malformed or inconsistent GDN tensors");
    c.conv_kernel_size = static_cast<int>(conv.shape[0]);
    c.gdn_num_v_heads = static_cast<int>(alpha.shape[1]);
    c.gdn_value_dim = static_cast<int>(gn.shape[0]);
    c.gdn_num_k_heads = l.get_meta_int("qwen35moe.ssm.group_count", 16);
    if (z.shape[1] != static_cast<int64_t>(c.gdn_num_v_heads) * c.gdn_value_dim || qkv.shape[1] < z.shape[1] ||
        (qkv.shape[1] - z.shape[1]) % 2 != 0)
        throw std::runtime_error("inconsistent GDN Q/K/V dimensions");
    const int64_t key_total = (qkv.shape[1] - z.shape[1]) / 2;
    if (c.gdn_num_k_heads <= 0 || key_total <= 0 || key_total % c.gdn_num_k_heads != 0)
        throw std::runtime_error("cannot derive GDN key dimensions");
    c.gdn_key_dim = static_cast<int>(key_total / c.gdn_num_k_heads);

    const int ai = c.full_attn_interval - 1;
    const std::string p = "blk." + std::to_string(ai) + ".";
    const auto& aq = tensor(l, p + "attn_q.weight");
    const auto& ak = tensor(l, p + "attn_k.weight");
    const auto& av = tensor(l, p + "attn_v.weight");
    const auto& qn = tensor(l, p + "attn_q_norm.weight");
    if (aq.shape.size()!=2 || ak.shape.size()!=2 || av.shape.size()!=2 || qn.shape.size()!=1 ||
        aq.shape[0]!=c.d_model || ak.shape[0]!=c.d_model || av.shape!=ak.shape || qn.shape[0]<=0)
        throw std::runtime_error("malformed or inconsistent full-attention tensors");
    c.head_dim = static_cast<int>(qn.shape[0]);
    if (aq.shape[1] % (2LL*c.head_dim) != 0 || ak.shape[1] % c.head_dim != 0)
        throw std::runtime_error("attention projection width is not head-aligned");
    c.num_heads = static_cast<int>(aq.shape[1] / (2LL*c.head_dim));
    c.num_kv_heads = static_cast<int>(ak.shape[1] / c.head_dim);
    c.partial_rope_dim = 64;
    if (const auto* sections = l.get_meta_array("qwen35moe.rope.dimension_sections")) {
        int64_t sum = 0;
        for (const auto& s : *sections) { const int part = std::stoi(s); if (part < 0) throw std::runtime_error("negative RoPE section"); sum += part; }
        if (sum > 0 && sum <= INT32_MAX/2) c.partial_rope_dim = static_cast<int>(2*sum);
    }

    const auto& eg = tensor(l, "blk.0.ffn_gate_exps.weight");
    const auto& sg = tensor(l, "blk.0.ffn_gate_shexp.weight");
    if (eg.shape.size()!=3 || sg.shape.size()!=2 || eg.shape[0]!=c.d_model || sg.shape[0]!=c.d_model)
        throw std::runtime_error("malformed MoE tensors");
    c.moe_intermediate_dim = static_cast<int>(eg.shape[1]);
    c.num_experts = static_cast<int>(eg.shape[2]);
    c.num_experts_per_tok = l.get_meta_int("qwen35moe.expert_used_count", 8);
    c.shared_expert_dim = static_cast<int>(sg.shape[1]);
    c.validate();
    return c;
}
std::string format_chat_prompt(const qwen::GgufLoader& loader,const qwen::BPETokenizer& tokenizer,
                               const std::string& user,const std::string& system) {
    const std::string chat_template=loader.get_meta_string("tokenizer.chat_template");
    if(chat_template.find("<|im_start|>")==std::string::npos||chat_template.find("<|im_end|>")==std::string::npos)
        throw std::runtime_error("--chat requires an embedded ChatML tokenizer.chat_template");
    if(tokenizer.token_id("<|im_start|>")<0||tokenizer.token_id("<|im_end|>")<0)
        throw std::runtime_error("ChatML special tokens are absent from the vocabulary");
    std::string out;
    if(!system.empty())out+="<|im_start|>system\n"+system+"<|im_end|>\n";
    out+="<|im_start|>user\n"+user+"<|im_end|>\n<|im_start|>assistant\n";
    return out;
}
std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open prompt file: " + path);
    std::ostringstream contents;
    contents << in.rdbuf();
    if (!in.good() && !in.eof()) throw std::runtime_error("cannot read prompt file: " + path);
    return contents.str();
}
double elapsed_ms(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}
struct VramObserver {
    bool enabled = false;
    size_t baseline_used = 0;
    size_t peak_used = 0;

    explicit VramObserver(bool should_observe) : enabled(should_observe) {
        if (!enabled) return;
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaFree(nullptr));
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        baseline_used = total_bytes - free_bytes;
        peak_used = baseline_used;
    }
    void sample() {
        if (!enabled) return;
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const size_t used = total_bytes - free_bytes;
        if (used > peak_used) peak_used = used;
    }
    double allocated_mib() const {
        return static_cast<double>(peak_used - baseline_used) / (1024.0 * 1024.0);
    }
};
void usage(const char*p){std::cout
 <<"Usage: "<<p<<" --weights MODEL.gguf [options]\n"
 <<"  --prompt TEXT              raw prompt or chat user message\n"
 <<"  --prompt-file PATH         read the prompt from a file\n"
 <<"  --chat                     apply embedded ChatML conversation format\n"
 <<"  --system TEXT              optional system message (requires --chat)\n"
 <<"  --max N                    maximum generated tokens (default 64)\n"
 <<"  --ctx N                    context length (default 4096)\n"
 <<"  --temperature F            0 for greedy, >0 for sampling (default 0)\n"
 <<"  --top-k N                  sampling candidate limit; 0 disables (default 40)\n"
 <<"  --top-p F                  nucleus threshold in (0,1] (default 0.95)\n"
 <<"  --repeat-penalty F         repeated-token penalty >=1 (default 1)\n"
 <<"  --repeat-last-n N          repetition history window (default 64)\n"
 <<"  --seed N                   deterministic sampling seed (default 0)\n"
 <<"  --benchmark                print detailed load and inference timings\n"
 <<"  --tokenize-only            print prompt token IDs without loading the GPU\n";}
}
int main(int argc,char**argv){std::string path,prompt="The capital of France is",prompt_file,system;int max_tokens=64,ctx_len=4096;bool tokenize_only=false,chat=false,benchmark=false,prompt_set=false;qwen::SamplingConfig sampling;try{for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--weights"&&i+1<argc)path=argv[++i];else if(a=="--prompt"&&i+1<argc){prompt=argv[++i];prompt_set=true;}else if(a=="--prompt-file"&&i+1<argc)prompt_file=argv[++i];else if(a=="--system"&&i+1<argc)system=argv[++i];else if(a=="--max"&&i+1<argc)max_tokens=std::stoi(argv[++i]);else if(a=="--ctx"&&i+1<argc)ctx_len=std::stoi(argv[++i]);else if(a=="--temperature"&&i+1<argc)sampling.temperature=std::stof(argv[++i]);else if(a=="--top-k"&&i+1<argc)sampling.top_k=std::stoi(argv[++i]);else if(a=="--top-p"&&i+1<argc)sampling.top_p=std::stof(argv[++i]);else if(a=="--repeat-penalty"&&i+1<argc)sampling.repetition_penalty=std::stof(argv[++i]);else if(a=="--repeat-last-n"&&i+1<argc)sampling.repetition_window=std::stoi(argv[++i]);else if(a=="--seed"&&i+1<argc)sampling.seed=std::stoull(argv[++i]);else if(a=="--chat")chat=true;else if(a=="--benchmark")benchmark=true;else if(a=="--tokenize-only")tokenize_only=true;else if(a=="--stream"){}else if(a=="--help"){usage(argv[0]);return 0;}else throw std::runtime_error("unknown or incomplete argument: "+a);}if(path.empty())throw std::runtime_error("--weights is required; there is no fake dry-run mode");if(prompt_set&&!prompt_file.empty())throw std::runtime_error("use either --prompt or --prompt-file, not both");if(!prompt_file.empty())prompt=read_text_file(prompt_file);if(prompt.empty())throw std::runtime_error("prompt must not be empty");if(!chat&&!system.empty())throw std::runtime_error("--system requires --chat");if(max_tokens<0||ctx_len<=0)throw std::runtime_error("invalid token/context limit");sampling.validate();
 const auto gguf_begin=std::chrono::steady_clock::now();qwen::GgufLoader loader;if(!loader.open(path))return 1;const auto gguf_end=std::chrono::steady_clock::now();qwen::ModelConfig cfg=config_from(loader,ctx_len);qwen::BPETokenizer tok;tok.init(loader);if(tok.vocab_size()!=cfg.vocab_size)throw std::runtime_error("tokenizer vocabulary and embedding vocabulary differ");const std::string model_prompt=chat?format_chat_prompt(loader,tok,prompt,system):prompt;auto ids=tok.encode(model_prompt,false,chat);if(ids.empty())throw std::runtime_error("prompt encoded to zero tokens");if(chat&&ids.front()!=tok.token_id("<|im_start|>"))throw std::runtime_error("ChatML control tokens were not parsed atomically; check tokenizer.ggml.token_type");if(tokenize_only){std::cout<<"TOKEN_IDS:";for(int id:ids)std::cout<<" "<<id;std::cout<<std::endl;return 0;}if(ids.size()+static_cast<size_t>(max_tokens)>static_cast<size_t>(ctx_len))throw std::runtime_error("prompt plus generation exceeds --ctx");VramObserver vram(benchmark);const auto tensor_begin=std::chrono::steady_clock::now();if(!loader.load_tensors_to_gpu(cfg.n_layers))throw std::runtime_error("failed to load required GGUF tensors to the GPU");const auto tensor_end=std::chrono::steady_clock::now();vram.sample();const auto init_begin=std::chrono::steady_clock::now();qwen::CudaContext cuda;qwen::QwenModel model;model.init(cfg,loader);qwen::Sampler sampler(sampling);const auto init_end=std::chrono::steady_clock::now();vram.sample();
 if(chat)std::cout<<"Assistant: "<<std::flush;else std::cout<<prompt<<std::flush;int pos=0;const float*logits=nullptr;const auto inference_begin=std::chrono::steady_clock::now();for(int id:ids)logits=model.decode_logits(id,pos++,cuda);const auto prefill_end=std::chrono::steady_clock::now();vram.sample();std::vector<int>history=ids;int generated=0,decode_steps=0;double decode_compute_ms=0.0;bool stopped_eog=false,first_token_seen=false;std::chrono::steady_clock::time_point first_token_time=prefill_end;while(generated<max_tokens){int next=sampler.sample(logits,model.vocab_size(),history);if(!first_token_seen){first_token_time=std::chrono::steady_clock::now();first_token_seen=true;}if(tok.is_eog(next)){stopped_eog=true;break;}std::cout<<tok.decode(next)<<std::flush;history.push_back(next);++generated;if(generated==max_tokens)break;const auto step_begin=std::chrono::steady_clock::now();logits=model.decode_logits(next,pos++,cuda);const auto step_end=std::chrono::steady_clock::now();decode_compute_ms+=elapsed_ms(step_begin,step_end);++decode_steps;}const auto inference_end=std::chrono::steady_clock::now();vram.sample();const double generation_sec=elapsed_ms(prefill_end,inference_end)/1000.0;std::cout<<"\n\nGenerated "<<generated<<" tokens in "<<generation_sec<<" s";if(generation_sec>0)std::cout<<" ("<<generated/generation_sec<<" tok/s)";std::cout<<" [stop: "<<(stopped_eog?"eog":"max-tokens")<<"]\n";
 if(benchmark){const double gguf_ms=elapsed_ms(gguf_begin,gguf_end);const double tensor_ms=elapsed_ms(tensor_begin,tensor_end);const double init_ms=elapsed_ms(init_begin,init_end);const double prefill_ms=elapsed_ms(inference_begin,prefill_end);const double decode_ms=elapsed_ms(prefill_end,inference_end);const double total_ms=elapsed_ms(inference_begin,inference_end);std::cout<<std::fixed<<std::setprecision(3)<<"\nBenchmark:\n"<<"gguf_open_ms: "<<gguf_ms<<"\n"<<"tensor_load_ms: "<<tensor_ms<<"\n"<<"model_init_ms: "<<init_ms<<"\n"<<"model_load_ms: "<<elapsed_ms(gguf_begin,init_end)<<"\n"<<"prompt_tokens: "<<ids.size()<<"\n"<<"prefill_ms: "<<prefill_ms<<"\n"<<"prefill_tok_per_s: "<<(prefill_ms>0?ids.size()*1000.0/prefill_ms:0.0)<<"\n"<<"time_to_first_token_ms: "<<(first_token_seen?elapsed_ms(inference_begin,first_token_time):0.0)<<"\n"<<"generated_tokens: "<<generated<<"\n"<<"decode_steps: "<<decode_steps<<"\n"<<"decode_ms: "<<decode_ms<<"\n"<<"decode_compute_ms: "<<decode_compute_ms<<"\n"<<"output_tok_per_s: "<<(decode_ms>0?generated*1000.0/decode_ms:0.0)<<"\n"<<"decode_step_tok_per_s: "<<(decode_compute_ms>0?decode_steps*1000.0/decode_compute_ms:0.0)<<"\n"<<"total_inference_ms: "<<total_ms<<"\n"<<"peak_observed_vram_mib: "<<vram.allocated_mib()<<"\n"<<"stop_reason: "<<(stopped_eog?"eog":"max-tokens")<<"\n";}return 0;
 }catch(const std::exception&e){std::cerr<<"Fatal: "<<e.what()<<std::endl;return 1;}}
