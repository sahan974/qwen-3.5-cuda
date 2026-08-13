#include "tokenizer.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace qwen {
namespace {

std::string utf8(uint32_t cp) {
    std::string out;
    if (cp < 0x80) out.push_back(static_cast<char>(cp));
    else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
    return out;
}

bool decode_cp(const std::string& s, size_t& i, uint32_t& cp) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) { cp = c; ++i; return true; }
    if ((c & 0xe0) == 0xc0 && i + 1 < s.size()) {
        cp = ((c & 0x1f) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3f); i += 2; return true;
    }
    if ((c & 0xf0) == 0xe0 && i + 2 < s.size()) {
        cp = ((c & 0x0f) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3f) << 6) |
             (static_cast<unsigned char>(s[i + 2]) & 0x3f); i += 3; return true;
    }
    return false;
}

bool letter(unsigned char c) { return std::isalpha(c) != 0 || c >= 0x80; }
bool digit(unsigned char c) { return c >= '0' && c <= '9'; }
bool space(unsigned char c) { return std::isspace(c) != 0; }

} // namespace

void BPETokenizer::build_byte_tables() {
    unicode_to_byte_.clear();
    std::array<bool, 256> used{};
    std::array<uint32_t, 256> cps{};
    for (int x = '!'; x <= '~'; ++x) { used[x] = true; cps[x] = x; }
    for (int x = 0xa1; x <= 0xac; ++x) { used[x] = true; cps[x] = x; }
    for (int x = 0xae; x <= 0xff; ++x) { used[x] = true; cps[x] = x; }
    uint32_t extra = 0;
    for (int x = 0; x < 256; ++x) if (!used[x]) cps[x] = 256 + extra++;
    for (int x = 0; x < 256; ++x) {
        byte_to_unicode_[x] = utf8(cps[x]);
        unicode_to_byte_[cps[x]] = static_cast<unsigned char>(x);
    }
}

bool BPETokenizer::init(const GgufLoader& loader) {
    vocab_.clear(); token_to_id_.clear(); bpe_ranks_.clear(); eog_ids_.clear();
    build_byte_tables();
    const auto* tokens = loader.get_meta_array("tokenizer.ggml.tokens");
    const auto* merges = loader.get_meta_array("tokenizer.ggml.merges");
    if (!tokens || tokens->empty()) throw std::runtime_error("GGUF is missing tokenizer.ggml.tokens");
    if (!merges || merges->empty()) throw std::runtime_error("GGUF is missing tokenizer.ggml.merges");
    vocab_ = *tokens;
    token_to_id_.reserve(vocab_.size() * 2);
    for (size_t i = 0; i < vocab_.size(); ++i) token_to_id_.emplace(vocab_[i], static_cast<int>(i));
    bpe_ranks_.reserve(merges->size() * 2);
    for (size_t i = 0; i < merges->size(); ++i) bpe_ranks_.emplace((*merges)[i], static_cast<int>(i));

    bos_token_id_ = loader.get_meta_int("tokenizer.ggml.bos_token_id", -1);
    for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
        int id = loader.get_meta_int(key, -1); if (id >= 0) eog_ids_.insert(id);
    }
    for (const char* special : {"<|endoftext|>", "<|im_end|>", "<|eot_id|>"}) {
        auto it = token_to_id_.find(special); if (it != token_to_id_.end()) eog_ids_.insert(it->second);
    }
    return true;
}

size_t BPETokenizer::next_chunk(const std::string& t, size_t i) const {
    const size_t n = t.size(); const unsigned char c = static_cast<unsigned char>(t[i]);
    if (c == '\'' && i + 1 < n) {
        for (const char* form : {"s", "t", "re", "ve", "m", "ll", "d"}) {
            size_t len = std::char_traits<char>::length(form);
            if (i + 1 + len <= n) {
                bool ok = true;
                for (size_t j = 0; j < len; ++j) ok &= std::tolower(static_cast<unsigned char>(t[i + 1 + j])) == form[j];
                if (ok) return i + 1 + len;
            }
        }
    }
    size_t j = i;
    if (!letter(c) && c != '\r' && c != '\n' && !digit(c) && i + 1 < n && letter(static_cast<unsigned char>(t[i + 1]))) j++;
    if (letter(static_cast<unsigned char>(t[j]))) { while (++j < n && letter(static_cast<unsigned char>(t[j]))) {} return j; }
    if (digit(c)) return i + 1;
    j = i;
    if (c == ' ' && i + 1 < n) {
        unsigned char d = static_cast<unsigned char>(t[i + 1]);
        if (!space(d) && !letter(d) && !digit(d)) ++j;
    }
    if (j < n) {
        unsigned char d = static_cast<unsigned char>(t[j]);
        if (!space(d) && !letter(d) && !digit(d)) {
            while (++j < n) { unsigned char e = static_cast<unsigned char>(t[j]); if (space(e) || letter(e) || digit(e)) break; }
            while (j < n && (t[j] == '\r' || t[j] == '\n')) ++j;
            return j;
        }
    }
    if (space(c)) { while (++j < n && space(static_cast<unsigned char>(t[j]))) {} return j; }
    return i + 1;
}

void BPETokenizer::encode_chunk(const std::string& chunk, std::vector<int>& out) const {
    std::vector<std::string> symbols; symbols.reserve(chunk.size());
    for (unsigned char c : chunk) symbols.push_back(byte_to_unicode_[c]);
    while (symbols.size() > 1) {
        int best_rank = 0x7fffffff; size_t best = symbols.size();
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = bpe_ranks_.find(symbols[i] + " " + symbols[i + 1]);
            if (it != bpe_ranks_.end() && it->second < best_rank) { best_rank = it->second; best = i; }
        }
        if (best == symbols.size()) break;
        symbols[best] += symbols[best + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1));
    }
    for (const auto& symbol : symbols) {
        auto it = token_to_id_.find(symbol);
        if (it == token_to_id_.end()) throw std::runtime_error("tokenizer produced a symbol absent from its vocabulary");
        out.push_back(it->second);
    }
}

std::vector<int> BPETokenizer::encode(const std::string& text, bool add_bos) const {
    if (vocab_.empty()) throw std::runtime_error("tokenizer is not initialized");
    std::vector<int> out;
    if (add_bos && bos_token_id_ >= 0) out.push_back(bos_token_id_);
    for (size_t i = 0; i < text.size();) { size_t end = next_chunk(text, i); encode_chunk(text.substr(i, end - i), out); i = end; }
    return out;
}

std::string BPETokenizer::decode(int id) const {
    if (id < 0 || id >= static_cast<int>(vocab_.size()) || eog_ids_.count(id)) return {};
    const std::string& token = vocab_[id];
    if (token.size() >= 3 && token.front() == '<' && token[1] == '|') return token;
    std::string out;
    for (size_t i = 0; i < token.size();) {
        uint32_t cp = 0; size_t before = i;
        if (!decode_cp(token, i, cp)) { out.push_back(token[before]); i = before + 1; continue; }
        auto it = unicode_to_byte_.find(cp);
        if (it == unicode_to_byte_.end()) out.append(token, before, i - before);
        else out.push_back(static_cast<char>(it->second));
    }
    return out;
}

std::string BPETokenizer::decode(const std::vector<int>& ids) const {
    std::string out; for (int id : ids) out += decode(id); return out;
}

} // namespace qwen
