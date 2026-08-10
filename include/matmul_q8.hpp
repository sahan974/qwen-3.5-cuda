#ifndef QWEN_MATMUL_Q8_HPP
#define QWEN_MATMUL_Q8_HPP

#include <cuda_runtime.h>
#include <cstdint>

namespace qwen {

// W8A16 MatMul: matrix A (M x K, int8) * vector x (K, fp16/fp32) -> vector y (M, fp32)
// Each row of A has an FP32 scale factor in scale_ptr
void matmul_w8a16(
    const int8_t* A_ptr,
    const float* scale_ptr,
    const float* x_ptr,
    float* y_ptr,
    int M,
    int K,
    cudaStream_t stream = nullptr
);

} // namespace qwen

#endif // QWEN_MATMUL_Q8_HPP
