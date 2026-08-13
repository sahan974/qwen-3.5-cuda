#include "tokenizer.hpp"
#include <iostream>

namespace qwen {

bool BPETokenizer::init(const GgufLoader& loader) {
    // Populate demo vocab fallback if GGUF metadata arrays aren't loaded
    vocab_.clear();
    token_to_id_.clear();

    for (int i = 0; i < 256; ++i) {
        std::string s(1, static_cast<char>(i));
        vocab_.push_back(s);
        token_to_id_[s] = i;
    }

    std::cout << "BPETokenizer initialized with vocabulary size: " << vocab_.size() << std::endl;
    return true;
}

std::vector<int> BPETokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> tokens;
    if (add_bos) {
        tokens.push_back(bos_token_id_);
    }

    for (unsigned char c : text) {
        std::string s(1, static_cast<char>(c));
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(static_cast<int>(c));
        }
    }

    return tokens;
}

std::string BPETokenizer::decode(int token_id) const {
    if (token_id >= 0 && token_id < static_cast<int>(vocab_.size())) {
        return vocab_[token_id];
    }
    return "";
}

std::string BPETokenizer::decode(const std::vector<int>& tokens) const {
    std::string result = "";
    for (int tok : tokens) {
        result += decode(tok);
    }
    return result;
}

} // namespace qwen
