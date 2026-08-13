#ifndef QWEN_LOADER_GGUF_HPP
#define QWEN_LOADER_GGUF_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include "quantized.hpp"

namespace qwen {

enum class GgufMetadataValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

class GgufLoader {
public:
    GgufLoader() = default;
    ~GgufLoader();

    bool open(const std::string& filepath);
    bool load_tensors_to_gpu();
    void unload_gpu();
    void close();

    uint32_t version() const { return version_; }
    uint64_t tensor_count() const { return tensor_count_; }
    uint64_t metadata_count() const { return metadata_count_; }
    uint64_t total_vram_bytes() const { return total_vram_bytes_; }
    
    const std::vector<QuantTensor>& tensors() const { return tensors_; }
    const std::unordered_map<std::string, std::string>& metadata() const { return metadata_str_; }
    const std::vector<std::string>* get_meta_array(const std::string& key) const {
        auto it = metadata_arrays_.find(key);
        return it == metadata_arrays_.end() ? nullptr : &it->second;
    }

    const QuantTensor* get_tensor(const std::string& name) const {
        for (const auto& t : tensors_) {
            if (t.name == name) return &t;
        }
        return nullptr;
    }


    // Get an integer metadata value by key (returns default_val if not found)
    int get_meta_int(const std::string& key, int default_val = 0) const {
        auto it = metadata_str_.find(key);
        if (it == metadata_str_.end()) return default_val;
        try { return std::stoi(it->second); } catch (...) { return default_val; }
    }
    double get_meta_float(const std::string& key, double default_val = 0.0) const {
        auto it = metadata_str_.find(key);
        if (it == metadata_str_.end()) return default_val;
        try { return std::stod(it->second); } catch (...) { return default_val; }
    }
    std::string get_meta_string(const std::string& key, const std::string& default_val = {}) const {
        auto it = metadata_str_.find(key);
        return it == metadata_str_.end() ? default_val : it->second;
    }

    void print_summary() const;

private:
    std::string filepath_;
    uint32_t version_ = 0;
    uint64_t tensor_count_ = 0;
    uint64_t metadata_count_ = 0;
    uint64_t payload_offset_ = 0;
    uint64_t total_vram_bytes_ = 0;
    
    std::vector<QuantTensor> tensors_;
    std::unordered_map<std::string, std::string> metadata_str_;
    std::unordered_map<std::string, std::vector<std::string>> metadata_arrays_;

    std::string read_string(std::ifstream& fin);
    void skip_metadata_value(std::ifstream& fin, GgufMetadataValueType vtype);
    std::string read_metadata_value_as_string(std::ifstream& fin, GgufMetadataValueType vtype);
    std::vector<std::string> read_metadata_array(std::ifstream& fin);
};

} // namespace qwen

#endif // QWEN_LOADER_GGUF_HPP
