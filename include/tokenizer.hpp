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
    bool init(const GgufLoader& loader);
    std::vector<int> encode(const std::string& text, bool add_bos = false) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& tokens) const;
    bool is_eog(int token_id) const { return eog_ids_.count(token_id) != 0; }
    int vocab_size() const { return static_cast<int>(vocab_.size()); }

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::string, int> bpe_ranks_;
    std::array<std::string, 256> byte_to_unicode_{};
    std::unordered_map<uint32_t, unsigned char> unicode_to_byte_;
    std::unordered_set<int> eog_ids_;
    int bos_token_id_ = -1;

    void build_byte_tables();
    void encode_chunk(const std::string& chunk, std::vector<int>& out) const;
};

} // namespace qwen

#endif
