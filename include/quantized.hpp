#ifndef QWEN_QUANTIZED_HPP
#define QWEN_QUANTIZED_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace qwen {

// GGUF Quantization Types enum (matching GGML format specs)
enum class GgmlType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    COUNT
};

struct QuantTensor {
    std::string name;
    GgmlType type;
    std::vector<int64_t> shape;
    uint64_t num_elements;
    uint64_t size_bytes;
    uint64_t offset;          // Offset in binary payload
    void* device_ptr = nullptr;// Device VRAM memory pointer
};

} // namespace qwen

#endif // QWEN_QUANTIZED_HPP
