#ifndef QWEN_TOKENIZER_HPP
#define QWEN_TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include "loader_gguf.hpp"

namespace qwen {

class BPETokenizer {
public:
    BPETokenizer() = default;

    // Load vocabulary and merges from GGUF metadata or default fallback
    bool init(const GgufLoader& loader);

    // Encode text prompt to token IDs
    std::vector<int> encode(const std::string& text, bool add_bos = true) const;

    // Decode token ID to string snippet
    std::string decode(int token_id) const;

    // Decode array of token IDs to full string
    std::string decode(const std::vector<int>& tokens) const;

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> bpe_ranks_;

    int bos_token_id_ = 151643; // Default Qwen3.5 BOS
    int eos_token_id_ = 151643; // Default Qwen3.5 EOS
};

} // namespace qwen

#endif // QWEN_TOKENIZER_HPP
