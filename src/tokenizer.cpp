#include "tokenizer.hpp"
#include "unicode-data.h"
#include <algorithm>
#include <cstdint>
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
    if ((c & 0xf8) == 0xf0 && i + 3 < s.size()) {
        cp = ((c & 0x07) << 18) | ((static_cast<unsigned char>(s[i + 1]) & 0x3f) << 12) |
             ((static_cast<unsigned char>(s[i + 2]) & 0x3f) << 6) |
             (static_cast<unsigned char>(s[i + 3]) & 0x3f); i += 4; return true;
    }
    return false;
}

enum : uint16_t { NUMBER = 0x0002, LETTER = 0x0004, ACCENT_MARK = 0x0010, WHITESPACE = 0x0100 };

uint16_t unicode_flags(uint32_t cp) {
    if (cp >= MAX_CODEPOINTS) return 0;
    const auto begin = unicode_ranges_flags.begin();
    const auto end = unicode_ranges_flags.end();
    auto it = std::upper_bound(begin, end, cp,
        [](uint32_t value, const std::pair<uint32_t, uint16_t>& range) { return value < range.first; });
    uint16_t flags = it == begin ? 0 : std::prev(it)->second;
    if (unicode_set_whitespace.count(cp)) flags |= WHITESPACE;
    return flags;
}

struct Codepoint {
    uint32_t value;
    size_t byte_begin;
    size_t byte_end;
    uint16_t flags;
};

std::vector<Codepoint> codepoints(const std::string& text) {
    std::vector<Codepoint> out;
    for (size_t i = 0; i < text.size();) {
        const size_t begin = i;
        uint32_t cp = 0;
        if (!decode_cp(text, i, cp) || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
            throw std::runtime_error("prompt contains invalid UTF-8");
        out.push_back({cp, begin, i, unicode_flags(cp)});
    }
    return out;
}

bool has(uint16_t flags, uint16_t bit) { return (flags & bit) != 0; }

// Exact custom pre-tokenizer used by Qwen3.5 in llama.cpp b9222. The returned
// offsets are UTF-8 byte offsets; BPE itself still operates on encoded bytes.
std::vector<size_t> qwen35_chunk_ends(const std::string& text) {
    const auto cps = codepoints(text);
    std::vector<size_t> ends;
    const auto value = [&](size_t p) { return p < cps.size() ? cps[p].value : UINT32_MAX; };
    const auto flags = [&](size_t p) { return p < cps.size() ? cps[p].flags : uint16_t{0}; };
    const auto add = [&](size_t p) { ends.push_back(p < cps.size() ? cps[p].byte_begin : text.size()); };

    for (size_t pos = 0; pos < cps.size();) {
        const uint32_t cp = value(pos);
        const uint16_t f = flags(pos);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp == '\'' && pos + 1 < cps.size()) {
            const auto lower_ascii = [](uint32_t x) { return x >= 'A' && x <= 'Z' ? x + ('a' - 'A') : x; };
            const uint32_t a = lower_ascii(value(pos + 1));
            if (a == 's' || a == 't' || a == 'm' || a == 'd') { pos += 2; add(pos); continue; }
            if (pos + 2 < cps.size()) {
                const uint32_t b = lower_ascii(value(pos + 2));
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
                    pos += 3; add(pos); continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (cp != '\r' && cp != '\n' && !has(f, NUMBER) &&
            (has(f, LETTER | ACCENT_MARK) || has(flags(pos + 1), LETTER | ACCENT_MARK))) {
            ++pos;
            while (has(flags(pos), LETTER | ACCENT_MARK)) ++pos;
            add(pos); continue;
        }

        // \p{N}
        if (has(f, NUMBER)) { ++pos; add(pos); continue; }

        // <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        uint16_t f2 = cp == ' ' ? flags(pos + 1) : f;
        const uint16_t excluded = WHITESPACE | LETTER | ACCENT_MARK | NUMBER;
        if (f != 0 && (f2 & excluded) == 0 && f2 != 0) {
            pos += cp == ' ';
            while (f2 != 0 && (f2 & excluded) == 0) f2 = flags(++pos);
            while (value(pos) == '\r' || value(pos) == '\n') ++pos;
            add(pos); continue;
        }

        size_t whitespace = 0;
        size_t last_newline_end = 0;
        while (has(flags(pos + whitespace), WHITESPACE)) {
            const uint32_t x = value(pos + whitespace);
            if (x == '\r' || x == '\n') last_newline_end = pos + whitespace + 1;
            ++whitespace;
        }
        if (last_newline_end != 0) { pos = last_newline_end; add(pos); continue; }
        if (whitespace > 1 && value(pos + whitespace) != UINT32_MAX) {
            pos += whitespace - 1; add(pos); continue;
        }
        if (whitespace > 0) { pos += whitespace; add(pos); continue; }

        ++pos;
        add(pos);
    }
    return ends;
}

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
    vocab_.clear(); token_to_id_.clear(); bpe_ranks_.clear(); eog_ids_.clear(); control_ids_.clear(); special_tokens_.clear();
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

    const auto* token_types = loader.get_meta_array("tokenizer.ggml.token_type");
    if (token_types && token_types->size() != vocab_.size())
        throw std::runtime_error("tokenizer.ggml.token_type size differs from vocabulary");
    if (token_types) {
        for (size_t i = 0; i < token_types->size(); ++i) {
            const int type = std::stoi((*token_types)[i]);
            // GGUF token types: UNKNOWN=2, CONTROL=3, USER_DEFINED=4.
            if ((type == 2 || type == 3 || type == 4) && !vocab_[i].empty()) special_tokens_.emplace_back(vocab_[i], static_cast<int>(i));
            if (type == 3) control_ids_.insert(static_cast<int>(i));
        }
        std::stable_sort(special_tokens_.begin(), special_tokens_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }

    bos_token_id_ = loader.get_meta_int("tokenizer.ggml.bos_token_id", -1);
    for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
        int id = loader.get_meta_int(key, -1); if (id >= 0) eog_ids_.insert(id);
    }
    for (const char* special : {"<|eot_id|>", "<|im_end|>", "<|end|>", "<|endoftext|>",
                                "<|eom_id|>", "<|end_of_text|>", "<end_of_turn>"}) {
        auto it = token_to_id_.find(special); if (it != token_to_id_.end()) eog_ids_.insert(it->second);
    }
    return true;
}

void BPETokenizer::encode_chunk(const std::string& chunk, std::vector<int>& out) const {
    std::vector<std::string> symbols; symbols.reserve(chunk.size());
    for (unsigned char c : chunk) symbols.push_back(byte_to_unicode_[c]);
    while (symbols.size() > 1) {
        int best_rank = 0x7fffffff;
        std::string best_left, best_right;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = bpe_ranks_.find(symbols[i] + " " + symbols[i + 1]);
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second; best_left = symbols[i]; best_right = symbols[i + 1];
            }
        }
        if (best_rank == 0x7fffffff) break;
        // GPT-2 BPE merges every non-overlapping occurrence of the selected
        // pair in one pass. Merging only the first occurrence can change which
        // lower-rank pair becomes available next and produce different IDs.
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size() && symbols[i] == best_left && symbols[i + 1] == best_right) {
                merged.push_back(symbols[i] + symbols[i + 1]); i += 2;
            } else {
                merged.push_back(std::move(symbols[i++]));
            }
        }
        symbols = std::move(merged);
    }
    for (const auto& symbol : symbols) {
        auto it = token_to_id_.find(symbol);
        if (it == token_to_id_.end()) throw std::runtime_error("tokenizer produced a symbol absent from its vocabulary");
        out.push_back(it->second);
    }
}

void BPETokenizer::encode_plain(const std::string& text, std::vector<int>& out) const {
    size_t begin = 0;
    for (size_t end : qwen35_chunk_ends(text)) {
        encode_chunk(text.substr(begin, end - begin), out);
        begin = end;
    }
}

std::vector<int> BPETokenizer::encode(const std::string& text, bool add_bos, bool parse_special) const {
    if (vocab_.empty()) throw std::runtime_error("tokenizer is not initialized");
    std::vector<int> out;
    if (add_bos && bos_token_id_ >= 0) out.push_back(bos_token_id_);
    if (!parse_special || special_tokens_.empty()) {
        encode_plain(text, out);
        return out;
    }
    size_t plain_begin = 0;
    for (size_t pos = 0; pos < text.size();) {
        int matched = -1;
        size_t matched_len = 0;
        for (const auto& special : special_tokens_) {
            if (special.first.size() <= text.size() - pos &&
                text.compare(pos, special.first.size(), special.first) == 0) {
                matched = special.second; matched_len = special.first.size(); break;
            }
        }
        if (matched < 0) { ++pos; continue; }
        if (pos > plain_begin) encode_plain(text.substr(plain_begin, pos - plain_begin), out);
        out.push_back(matched);
        pos += matched_len;
        plain_begin = pos;
    }
    if (plain_begin < text.size()) encode_plain(text.substr(plain_begin), out);
    return out;
}

int BPETokenizer::token_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return it == token_to_id_.end() ? -1 : it->second;
}

std::string BPETokenizer::decode(int id) const {
    if (id < 0 || id >= static_cast<int>(vocab_.size()) || eog_ids_.count(id) || control_ids_.count(id)) return {};
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
