#include "config.hpp"
#include "cuda_utils.hpp"
#include "loader_gguf.hpp"
#include "model.hpp"
#include "tokenizer.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
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
void usage(const char*p){std::cout<<"Usage: "<<p<<" --weights MODEL.gguf [--prompt TEXT] [--max N] [--ctx N] [--tokenize-only]\n";}
}
int main(int argc,char**argv){std::string path,prompt="The capital of France is";int max_tokens=64,ctx_len=4096;bool tokenize_only=false;try{for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--weights"&&i+1<argc)path=argv[++i];else if(a=="--prompt"&&i+1<argc)prompt=argv[++i];else if(a=="--max"&&i+1<argc)max_tokens=std::stoi(argv[++i]);else if(a=="--ctx"&&i+1<argc)ctx_len=std::stoi(argv[++i]);else if(a=="--tokenize-only")tokenize_only=true;else if(a=="--stream"){}else if(a=="--help"){usage(argv[0]);return 0;}else throw std::runtime_error("unknown or incomplete argument: "+a);}if(path.empty())throw std::runtime_error("--weights is required; there is no fake dry-run mode");if(prompt.empty())throw std::runtime_error("prompt must not be empty");if(max_tokens<0||ctx_len<=0)throw std::runtime_error("invalid token/context limit");
 qwen::GgufLoader loader;if(!loader.open(path))return 1;qwen::ModelConfig cfg=config_from(loader,ctx_len);qwen::BPETokenizer tok;tok.init(loader);if(tok.vocab_size()!=cfg.vocab_size)throw std::runtime_error("tokenizer vocabulary and embedding vocabulary differ");auto ids=tok.encode(prompt,false);if(ids.empty())throw std::runtime_error("prompt encoded to zero tokens");if(tokenize_only){std::cout<<"TOKEN_IDS:";for(int id:ids)std::cout<<" "<<id;std::cout<<std::endl;return 0;}if(ids.size()+static_cast<size_t>(max_tokens)>static_cast<size_t>(ctx_len))throw std::runtime_error("prompt plus generation exceeds --ctx");if(!loader.load_tensors_to_gpu(cfg.n_layers))throw std::runtime_error("failed to load required GGUF tensors to the GPU");qwen::CudaContext cuda;qwen::QwenModel model;model.init(cfg,loader);
 std::cout<<prompt<<std::flush;int pos=0,next=-1;for(int id:ids)next=model.decode_step(id,pos++,cuda);auto start=std::chrono::steady_clock::now();int generated=0;while(generated<max_tokens&&!tok.is_eog(next)){std::cout<<tok.decode(next)<<std::flush;++generated;if(generated==max_tokens)break;next=model.decode_step(next,pos++,cuda);}double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::cout<<"\n\nGenerated "<<generated<<" tokens in "<<sec<<" s";if(sec>0)std::cout<<" ("<<generated/sec<<" tok/s)";std::cout<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<"Fatal: "<<e.what()<<std::endl;return 1;}}
