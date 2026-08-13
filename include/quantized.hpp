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
    F64     = 28,
    BF16    = 30,
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

// Returns block size (number of elements per block)
inline uint64_t ggml_type_block_size(GgmlType type) {
    switch (type) {
        case GgmlType::F32:     return 1;
        case GgmlType::F16:     return 1;
        case GgmlType::BF16:    return 1;
        case GgmlType::Q4_0:    return 32;
        case GgmlType::Q4_1:    return 32;
        case GgmlType::Q5_0:    return 32;
        case GgmlType::Q5_1:    return 32;
        case GgmlType::Q8_0:    return 32;
        case GgmlType::Q8_1:    return 32;
        case GgmlType::Q2_K:    return 256;
        case GgmlType::Q3_K:    return 256;
        case GgmlType::Q4_K:    return 256;
        case GgmlType::Q5_K:    return 256;
        case GgmlType::Q6_K:    return 256;
        case GgmlType::Q8_K:    return 256;
        case GgmlType::I8:      return 1;
        case GgmlType::I16:     return 1;
        case GgmlType::I32:     return 1;
        default:                return 0;
    }
}

// Returns block type size in bytes
inline uint64_t ggml_type_size(GgmlType type) {
    switch (type) {
        case GgmlType::F32:     return 4;
        case GgmlType::F16:     return 2;
        case GgmlType::BF16:    return 2;
        case GgmlType::Q4_0:    return 18;  // 2 bytes float16 scale + 16 bytes nibbles
        case GgmlType::Q4_1:    return 20;  // 2 bytes scale + 2 bytes min + 16 bytes nibbles
        case GgmlType::Q5_0:    return 22;
        case GgmlType::Q5_1:    return 24;
        case GgmlType::Q8_0:    return 34;  // 2 bytes float16 scale + 32 bytes int8
        case GgmlType::Q8_1:    return 36;
        case GgmlType::Q2_K:    return 82;
        case GgmlType::Q3_K:    return 110;
        case GgmlType::Q4_K:    return 144; // Standard Q4_K block size
        case GgmlType::Q5_K:    return 176;
        case GgmlType::Q6_K:    return 210;
        case GgmlType::Q8_K:    return 292;
        case GgmlType::I8:      return 1;
        case GgmlType::I16:     return 2;
        case GgmlType::I32:     return 4;
        default:                return 0;
    }
}

inline uint64_t calculate_tensor_bytes(GgmlType type, uint64_t num_elements) {
    uint64_t bs = ggml_type_block_size(type);
    uint64_t ts = ggml_type_size(type);
    if (bs == 0 || num_elements % bs != 0) return 0;
    return (num_elements / bs) * ts;
}

} // namespace qwen

#endif // QWEN_QUANTIZED_HPP
