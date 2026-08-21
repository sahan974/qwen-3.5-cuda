#ifndef QWEN_TOKENIZER_HPP
#define QWEN_TOKENIZER_HPP

#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "loader_gguf.hpp"

namespace qwen {

class BPETokenizer {
public:
    // --- Lifecycle ---
    bool init(const GgufLoader& loader);

    // --- Encoding & Decoding ---
    std::vector<int> encode(
        const std::string& text,
        bool               add_bos       = false,
        bool               parse_special = false
    ) const;

    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& tokens) const;

    // --- State & Accessors ---
    bool is_eog(int token_id) const { return eog_ids_.count(token_id) != 0; }
    int  vocab_size()         const { return static_cast<int>(vocab_.size()); }
    int  token_id(const std::string& token) const;

private:
    // --- Vocabulary Data ---
    std::vector<std::string>             vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> bpe_ranks_;

    // --- Byte-level BPE Lookup Tables ---
    std::array<std::string, 256>                byte_to_unicode_{};
    std::unordered_map<uint32_t, unsigned char> unicode_to_byte_;

    // --- Special Token Management ---
    std::unordered_set<int>                  eog_ids_;
    std::unordered_set<int>                  control_ids_;
    std::vector<std::pair<std::string, int>> special_tokens_;
    int bos_token_id_ = -1;

    // --- Internal Helpers ---
    void build_byte_tables();
    void encode_plain(const std::string& text, std::vector<int>& out) const;
    void encode_chunk(const std::string& chunk, std::vector<int>& out) const;
};

} // namespace qwen

#endif // QWEN_TOKENIZER_HPP
