#include "loader_gguf.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <limits>
#include <unordered_set>

namespace qwen {

static constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian

template <typename T>
T read_scalar(std::ifstream& fin, const char* what) {
    T value{};
    fin.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!fin) throw std::runtime_error(std::string("truncated GGUF while reading ") + what);
    return value;
}

int tensor_block_index(const std::string& name) {
    if (name.rfind("blk.", 0) != 0) return -1;
    size_t end = name.find('.', 4);
    if (end == std::string::npos || end == 4) return -1;
    int index = 0;
    for (size_t i = 4; i < end; ++i) {
        if (name[i] < '0' || name[i] > '9') return -1;
        if (index > (std::numeric_limits<int>::max() - (name[i] - '0')) / 10) return -1;
        index = index * 10 + (name[i] - '0');
    }
    return index;
}

GgufLoader::~GgufLoader() {
    close();
}

void GgufLoader::close() {
    unload_gpu();
    tensors_.clear();
    metadata_str_.clear();
    metadata_arrays_.clear();
}

std::string GgufLoader::read_string(std::ifstream& fin) {
    const uint64_t len = read_scalar<uint64_t>(fin, "string length");
    if (len > (1ULL << 32)) throw std::runtime_error("unreasonable GGUF string length");
    std::string str(len, '\0');
    if (len != 0) fin.read(str.data(), static_cast<std::streamsize>(len));
    if (!fin) throw std::runtime_error("truncated GGUF string");
    return str;
}

std::vector<std::string> GgufLoader::read_metadata_array(std::ifstream& fin) {
    const uint32_t raw_type = read_scalar<uint32_t>(fin, "array element type");
    const uint64_t count = read_scalar<uint64_t>(fin, "array length");
    if (count > (1ULL << 32)) throw std::runtime_error("invalid GGUF metadata array");
    const auto type = static_cast<GgufMetadataValueType>(raw_type);
    if (type == GgufMetadataValueType::ARRAY) throw std::runtime_error("nested GGUF arrays are unsupported");
    std::vector<std::string> result;
    result.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) result.push_back(read_metadata_value_as_string(fin, type));
    return result;
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
            const uint32_t raw_arr_type = read_scalar<uint32_t>(fin, "array element type");
            const uint64_t arr_len = read_scalar<uint64_t>(fin, "array length");
            if (arr_len > (1ULL << 32)) throw std::runtime_error("invalid GGUF metadata array length");
            GgufMetadataValueType arr_type = static_cast<GgufMetadataValueType>(raw_arr_type);
            for (uint64_t i = 0; i < arr_len; ++i) {
                skip_metadata_value(fin, arr_type);
            }
            break;
        }
        default:
            throw std::runtime_error("Unknown GGUF metadata value type");
    }
    if (!fin) throw std::runtime_error("truncated GGUF metadata value");
}

std::string GgufLoader::read_metadata_value_as_string(std::ifstream& fin, GgufMetadataValueType vtype) {
    switch (vtype) {
        case GgufMetadataValueType::UINT8: return std::to_string(read_scalar<uint8_t>(fin, "uint8 metadata"));
        case GgufMetadataValueType::INT8: return std::to_string(read_scalar<int8_t>(fin, "int8 metadata"));
        case GgufMetadataValueType::BOOL: return read_scalar<uint8_t>(fin, "bool metadata") ? "true" : "false";
        case GgufMetadataValueType::UINT16: return std::to_string(read_scalar<uint16_t>(fin, "uint16 metadata"));
        case GgufMetadataValueType::INT16: return std::to_string(read_scalar<int16_t>(fin, "int16 metadata"));
        case GgufMetadataValueType::UINT32: return std::to_string(read_scalar<uint32_t>(fin, "uint32 metadata"));
        case GgufMetadataValueType::INT32: return std::to_string(read_scalar<int32_t>(fin, "int32 metadata"));
        case GgufMetadataValueType::FLOAT32: return std::to_string(read_scalar<float>(fin, "float32 metadata"));
        case GgufMetadataValueType::UINT64: return std::to_string(read_scalar<uint64_t>(fin, "uint64 metadata"));
        case GgufMetadataValueType::INT64: return std::to_string(read_scalar<int64_t>(fin, "int64 metadata"));
        case GgufMetadataValueType::FLOAT64: return std::to_string(read_scalar<double>(fin, "float64 metadata"));
        case GgufMetadataValueType::STRING: return read_string(fin);
        case GgufMetadataValueType::ARRAY: {
            throw std::runtime_error("array metadata must be read with read_metadata_array");
        }
        default:
            skip_metadata_value(fin, vtype);
            return "<unsupported>";
    }
}

bool GgufLoader::open(const std::string& filepath) {
    close();
    filepath_ = filepath;
    std::ifstream fin(filepath, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Failed to open GGUF file: " << filepath << std::endl;
        return false;
    }

    const uint32_t magic = read_scalar<uint32_t>(fin, "magic");
    if (magic != GGUF_MAGIC) {
        std::cerr << "Invalid GGUF magic header in file: " << filepath << std::endl;
        return false;
    }

    version_ = read_scalar<uint32_t>(fin, "version");
    if (version_ != 2 && version_ != 3) throw std::runtime_error("unsupported GGUF version " + std::to_string(version_));
    tensor_count_ = read_scalar<uint64_t>(fin, "tensor count");
    metadata_count_ = read_scalar<uint64_t>(fin, "metadata count");
    if (tensor_count_ > (1ULL << 24) || metadata_count_ > (1ULL << 24))
        throw std::runtime_error("unreasonable GGUF header counts");

    // Parse metadata
    for (uint64_t i = 0; i < metadata_count_; ++i) {
        std::string key = read_string(fin);
        if (metadata_str_.count(key) || metadata_arrays_.count(key))
            throw std::runtime_error("duplicate GGUF metadata key: " + key);
        const uint32_t raw_vtype = read_scalar<uint32_t>(fin, "metadata value type");
        GgufMetadataValueType vtype = static_cast<GgufMetadataValueType>(raw_vtype);
        if (vtype == GgufMetadataValueType::ARRAY) metadata_arrays_[key] = read_metadata_array(fin);
        else metadata_str_[key] = read_metadata_value_as_string(fin, vtype);
    }

    // Parse tensor info headers
    tensors_.reserve(tensor_count_);
    std::unordered_set<std::string> names;
    for (uint64_t i = 0; i < tensor_count_; ++i) {
        QuantTensor tensor;
        tensor.name = read_string(fin);
        if (!names.insert(tensor.name).second) throw std::runtime_error("duplicate GGUF tensor name: " + tensor.name);
        
        const uint32_t n_dimensions = read_scalar<uint32_t>(fin, "tensor dimension count");
        if (n_dimensions == 0 || n_dimensions > 4) throw std::runtime_error("invalid dimensions for tensor " + tensor.name);
        
        tensor.shape.resize(n_dimensions);
        tensor.num_elements = 1;
        for (uint32_t d = 0; d < n_dimensions; ++d) {
            tensor.shape[d] = read_scalar<int64_t>(fin, "tensor dimension");
            if (tensor.shape[d] <= 0 || tensor.num_elements > std::numeric_limits<uint64_t>::max() / tensor.shape[d])
                throw std::runtime_error("invalid or overflowing shape for tensor " + tensor.name);
            tensor.num_elements *= static_cast<uint64_t>(tensor.shape[d]);
        }

        const uint32_t raw_type = read_scalar<uint32_t>(fin, "tensor type");
        tensor.type = static_cast<GgmlType>(raw_type);

        tensor.offset = read_scalar<uint64_t>(fin, "tensor offset");

        tensor.size_bytes = calculate_tensor_bytes(tensor.type, tensor.num_elements);
        if (tensor.size_bytes == 0) {
            throw std::runtime_error("unsupported or malformed tensor '" + tensor.name + "' (GGML type " +
                                     std::to_string(raw_type) + ")");
        }
        const uint64_t block = ggml_type_block_size(tensor.type);
        if (block > 1 && static_cast<uint64_t>(tensor.shape[0]) % block != 0)
            throw std::runtime_error("quantized row width is not block-aligned for tensor " + tensor.name);

        tensors_.push_back(tensor);
    }

    const uint64_t alignment = static_cast<uint64_t>(get_meta_int("general.alignment", 32));
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) throw std::runtime_error("invalid GGUF alignment");
    uint64_t cur_pos = fin.tellg();
    payload_offset_ = (cur_pos + alignment - 1) & ~(alignment - 1);

    fin.seekg(0, std::ios::end);
    const auto end_pos = fin.tellg();
    if (end_pos < 0) throw std::runtime_error("could not determine GGUF file size");
    const uint64_t file_size = static_cast<uint64_t>(end_pos);
    struct Span { uint64_t begin, end; const std::string* name; };
    std::vector<Span> spans;
    spans.reserve(tensors_.size());
    for (const auto& t : tensors_) {
        if (t.offset % alignment != 0 || t.offset > std::numeric_limits<uint64_t>::max() - payload_offset_ ||
            payload_offset_ + t.offset > file_size || t.size_bytes > file_size - (payload_offset_ + t.offset))
            throw std::runtime_error("invalid data range for tensor " + t.name);
        spans.push_back({payload_offset_ + t.offset, payload_offset_ + t.offset + t.size_bytes, &t.name});
    }
    std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) { return a.begin < b.begin; });
    for (size_t i = 1; i < spans.size(); ++i) {
        if (spans[i].begin < spans[i - 1].end)
            throw std::runtime_error("overlapping GGUF tensors: " + *spans[i - 1].name + " and " + *spans[i].name);
    }

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

bool GgufLoader::load_tensors_to_gpu(int base_layer_count) {
    std::ifstream fin(filepath_, std::ios::binary);
    if (!fin.is_open()) return false;

    unload_gpu();
    std::vector<char> buffer;
    size_t selected = 0;
    for (const auto& t : tensors_) {
        const int block = tensor_block_index(t.name);
        if (base_layer_count < 0 || block < 0 || block < base_layer_count) ++selected;
    }

    std::cout << "Loading " << selected << " of " << tensor_count_ << " tensors into GPU VRAM..." << std::endl;

    for (size_t i = 0; i < tensors_.size(); ++i) {
        auto& t = tensors_[i];
        const int block = tensor_block_index(t.name);
        if (base_layer_count >= 0 && block >= base_layer_count) continue;
        
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

        fin.read(buffer.data(), static_cast<std::streamsize>(t.size_bytes));
        if (!fin) {
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

    std::cout << "Successfully loaded required tensors. Total VRAM allocated: "
              << (total_vram_bytes_ / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    return true;
}

void GgufLoader::print_summary() const {
    std::cout << "--- GGUF Header Summary ---" << std::endl;
    std::cout << "File: "             << filepath_       << std::endl;
    std::cout << "GGUF Version: "    << version_        << std::endl;
    std::cout << "Tensor Count: "    << tensor_count_   << std::endl;
    std::cout << "Metadata KV Count: " << metadata_count_ << std::endl;

    std::cout << "\nAll Metadata Entries:" << std::endl;
    for (const auto& item : metadata_str_) {
        const std::string& k = item.first;
        const std::string& v = item.second;
        // Skip huge array values (tokenizer vocab etc.)
        if (v.size() > 120) {
            std::cout << "  " << k << " = [... " << v.size() << " chars ...]" << std::endl;
        } else {
            std::cout << "  " << k << " = " << v << std::endl;
        }
    }
}

} // namespace qwen
