#include "loader_gguf.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace qwen {

static constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian

GgufLoader::~GgufLoader() {
    close();
}

void GgufLoader::close() {
    tensors_.clear();
    metadata_str_.clear();
}

std::string GgufLoader::read_string(std::ifstream& fin) {
    uint64_t len = 0;
    fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!fin.good()) return "";
    std::string str(len, '\0');
    fin.read(&str[0], len);
    return str;
}

void GgufLoader::skip_metadata_value(std::ifstream& fin, GgufMetadataValueType vtype) {
    switch (vtype) {
        case GgufMetadataValueType::UINT8:
        case GgufMetadataValueType::INT8:
        case GgufMetadataValueType::BOOL:
            fin.seekg(1, std::ios::cur); break;
        case GgufMetadataValueType::UINT16:
        case GgufMetadataValueType::INT16:
            fin.seekg(2, std::ios::cur); break;
        case GgufMetadataValueType::UINT32:
        case GgufMetadataValueType::INT32:
        case GgufMetadataValueType::FLOAT32:
            fin.seekg(4, std::ios::cur); break;
        case GgufMetadataValueType::UINT64:
        case GgufMetadataValueType::INT64:
        case GgufMetadataValueType::FLOAT64:
            fin.seekg(8, std::ios::cur); break;
        case GgufMetadataValueType::STRING:
            read_string(fin); break;
        case GgufMetadataValueType::ARRAY: {
            uint32_t raw_arr_type = 0;
            uint64_t arr_len = 0;
            fin.read(reinterpret_cast<char*>(&raw_arr_type), sizeof(raw_arr_type));
            fin.read(reinterpret_cast<char*>(&arr_len), sizeof(arr_len));
            GgufMetadataValueType arr_type = static_cast<GgufMetadataValueType>(raw_arr_type);
            for (uint64_t i = 0; i < arr_len; ++i) {
                skip_metadata_value(fin, arr_type);
            }
            break;
        }
        default:
            throw std::runtime_error("Unknown GGUF metadata value type");
    }
}

std::string GgufLoader::read_metadata_value_as_string(std::ifstream& fin, GgufMetadataValueType vtype) {
    switch (vtype) {
        case GgufMetadataValueType::UINT8: { uint8_t v; fin.read(reinterpret_cast<char*>(&v), 1); return std::to_string(v); }
        case GgufMetadataValueType::INT8: { int8_t v; fin.read(reinterpret_cast<char*>(&v), 1); return std::to_string(v); }
        case GgufMetadataValueType::BOOL: { uint8_t v; fin.read(reinterpret_cast<char*>(&v), 1); return v ? "true" : "false"; }
        case GgufMetadataValueType::UINT16: { uint16_t v; fin.read(reinterpret_cast<char*>(&v), 2); return std::to_string(v); }
        case GgufMetadataValueType::INT16: { int16_t v; fin.read(reinterpret_cast<char*>(&v), 2); return std::to_string(v); }
        case GgufMetadataValueType::UINT32: { uint32_t v; fin.read(reinterpret_cast<char*>(&v), 4); return std::to_string(v); }
        case GgufMetadataValueType::INT32: { int32_t v; fin.read(reinterpret_cast<char*>(&v), 4); return std::to_string(v); }
        case GgufMetadataValueType::FLOAT32: { float v; fin.read(reinterpret_cast<char*>(&v), 4); return std::to_string(v); }
        case GgufMetadataValueType::UINT64: { uint64_t v; fin.read(reinterpret_cast<char*>(&v), 8); return std::to_string(v); }
        case GgufMetadataValueType::INT64: { int64_t v; fin.read(reinterpret_cast<char*>(&v), 8); return std::to_string(v); }
        case GgufMetadataValueType::FLOAT64: { double v; fin.read(reinterpret_cast<char*>(&v), 8); return std::to_string(v); }
        case GgufMetadataValueType::STRING: return read_string(fin);
        case GgufMetadataValueType::ARRAY: {
            uint32_t raw_arr_type = 0;
            uint64_t arr_len = 0;
            fin.read(reinterpret_cast<char*>(&raw_arr_type), sizeof(raw_arr_type));
            fin.read(reinterpret_cast<char*>(&arr_len), sizeof(arr_len));
            GgufMetadataValueType arr_type = static_cast<GgufMetadataValueType>(raw_arr_type);
            std::string res = "[";
            for (uint64_t i = 0; i < arr_len; ++i) {
                if (i > 0) res += ", ";
                res += read_metadata_value_as_string(fin, arr_type);
            }
            res += "]";
            return res;
        }
        default:
            skip_metadata_value(fin, vtype);
            return "<unsupported>";
    }
}

bool GgufLoader::open(const std::string& filepath) {
    filepath_ = filepath;
    std::ifstream fin(filepath, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Failed to open GGUF file: " << filepath << std::endl;
        return false;
    }

    uint32_t magic = 0;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != GGUF_MAGIC) {
        std::cerr << "Invalid GGUF magic header in file: " << filepath << std::endl;
        return false;
    }

    fin.read(reinterpret_cast<char*>(&version_), sizeof(version_));
    fin.read(reinterpret_cast<char*>(&tensor_count_), sizeof(tensor_count_));
    fin.read(reinterpret_cast<char*>(&metadata_count_), sizeof(metadata_count_));

    // Parse metadata
    for (uint64_t i = 0; i < metadata_count_; ++i) {
        std::string key = read_string(fin);
        uint32_t raw_vtype = 0;
        fin.read(reinterpret_cast<char*>(&raw_vtype), sizeof(raw_vtype));
        GgufMetadataValueType vtype = static_cast<GgufMetadataValueType>(raw_vtype);
        std::string val_str = read_metadata_value_as_string(fin, vtype);
        metadata_str_[key] = val_str;
    }

    // Parse tensor info headers
    tensors_.reserve(tensor_count_);
    for (uint64_t i = 0; i < tensor_count_; ++i) {
        QuantTensor tensor;
        tensor.name = read_string(fin);
        
        uint32_t n_dimensions = 0;
        fin.read(reinterpret_cast<char*>(&n_dimensions), sizeof(n_dimensions));
        
        tensor.shape.resize(n_dimensions);
        tensor.num_elements = 1;
        for (uint32_t d = 0; d < n_dimensions; ++d) {
            fin.read(reinterpret_cast<char*>(&tensor.shape[d]), sizeof(int64_t));
            tensor.num_elements *= tensor.shape[d];
        }

        uint32_t raw_type = 0;
        fin.read(reinterpret_cast<char*>(&raw_type), sizeof(raw_type));
        tensor.type = static_cast<GgmlType>(raw_type);

        fin.read(reinterpret_cast<char*>(&tensor.offset), sizeof(tensor.offset));

        tensor.size_bytes = calculate_tensor_bytes(tensor.type, tensor.num_elements);

        tensors_.push_back(tensor);
    }

    // Align to 32 bytes for binary payload start
    uint64_t cur_pos = fin.tellg();
    payload_offset_ = (cur_pos + 31) & ~31;

    return true;
}

void GgufLoader::unload_gpu() {
    for (auto& t : tensors_) {
        if (t.device_ptr != nullptr) {
            cudaFree(t.device_ptr);
            t.device_ptr = nullptr;
        }
    }
    total_vram_bytes_ = 0;
}

bool GgufLoader::load_tensors_to_gpu() {
    std::ifstream fin(filepath_, std::ios::binary);
    if (!fin.is_open()) return false;

    total_vram_bytes_ = 0;
    std::vector<char> buffer;

    std::cout << "Loading " << tensor_count_ << " tensors into GPU VRAM..." << std::endl;

    for (size_t i = 0; i < tensors_.size(); ++i) {
        auto& t = tensors_[i];
        
        // Allocate device VRAM
        cudaError_t err = cudaMalloc(&t.device_ptr, t.size_bytes);
        if (err != cudaSuccess) {
            std::cerr << "CUDA OOM allocating tensor " << t.name << " (" << t.size_bytes << " bytes)" << std::endl;
            unload_gpu();
            return false;
        }

        // Seek to absolute tensor offset in file payload
        fin.seekg(payload_offset_ + t.offset, std::ios::beg);
        
        if (buffer.size() < t.size_bytes) {
            buffer.resize(t.size_bytes);
        }

        fin.read(buffer.data(), t.size_bytes);
        if (!fin.good()) {
            std::cerr << "Error reading binary tensor data for: " << t.name << std::endl;
            unload_gpu();
            return false;
        }

        // Copy to GPU VRAM
        err = cudaMemcpy(t.device_ptr, buffer.data(), t.size_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpyHostToDevice failed for: " << t.name << std::endl;
            unload_gpu();
            return false;
        }

        total_vram_bytes_ += t.size_bytes;
    }

    std::cout << "Successfully loaded all tensors. Total VRAM allocated: " 
              << (total_vram_bytes_ / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    return true;
}

void GgufLoader::print_summary() const {
    std::cout << "--- GGUF Header Summary ---" << std::endl;
    std::cout << "File: " << filepath_ << std::endl;
    std::cout << "GGUF Version: " << version_ << std::endl;
    std::cout << "Tensor Count: " << tensor_count_ << std::endl;
    std::cout << "Metadata KV Count: " << metadata_count_ << std::endl;
    
    std::cout << "\nKey Metadata Entries:" << std::endl;
    for (const auto& [k, v] : metadata_str_) {
        if (k.find("architecture") != std::string::npos ||
            k.find("context_length") != std::string::npos ||
            k.find("embedding_length") != std::string::npos ||
            k.find("block_count") != std::string::npos) {
            std::cout << "  " << k << " = " << v << std::endl;
        }
    }
}

} // namespace qwen
