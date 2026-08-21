#ifndef QWEN_MATMUL_Q4_HPP
#define QWEN_MATMUL_Q4_HPP

#include <cuda_runtime.h>
#include "quantized.hpp"

namespace qwen {

// --- Compatibility Check ---
bool matmul_type_supported(GgmlType type);

// --- Matrix Multiplication ---
void matmul_dispatch(
    const QuantTensor& weight,
    const float*       x,
    float*             y,
    int                M,
    int                K,
    cudaStream_t       stream         = nullptr,
    uint64_t           element_offset = 0
);

// --- Dequantization Utility ---
void dequantize_row(
    const QuantTensor& tensor,
    float*             out,
    int                row,
    int                cols,
    cudaStream_t       stream = nullptr
);

} // namespace qwen

#endif // QWEN_MATMUL_Q4_HPP
