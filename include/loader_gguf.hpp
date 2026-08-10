#ifndef QWEN_LOADER_GGUF_HPP
#define QWEN_LOADER_GGUF_HPP

#include <string>
#include <vector>
#include <unordered_map>
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

    std::string read_string(std::ifstream& fin);
    void skip_metadata_value(std::ifstream& fin, GgufMetadataValueType vtype);
    std::string read_metadata_value_as_string(std::ifstream& fin, GgufMetadataValueType vtype);
};

} // namespace qwen

#endif // QWEN_LOADER_GGUF_HPP
