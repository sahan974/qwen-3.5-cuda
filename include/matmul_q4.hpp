#ifndef QWEN_MATMUL_Q4_HPP
#define QWEN_MATMUL_Q4_HPP

#include <cuda_runtime.h>
#include <cstdint>

namespace qwen {

// W4A16 MatMul: packed uint8 A (M x (K/2)), per-group scale (M x (K/group_size)), per-group zero_point
// Activations x (K, fp32), Output y (M, fp32)
void matmul_w4a16(
    const uint8_t* A_packed_ptr,
    const float* scales_ptr,
    const float* zeros_ptr,
    const float* x_ptr,
    float* y_ptr,
    int M,
    int K,
    int group_size = 32,
    cudaStream_t stream = nullptr
);

} // namespace qwen

#endif // QWEN_MATMUL_Q4_HPP
