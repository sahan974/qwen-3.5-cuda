#include "matmul_q4.hpp"
#include <cuda_fp16.h>
#include <stdio.h>

namespace qwen {

__device__ __forceinline__ float warp_reduce_sum_q4(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// INT4 W4A16 Dequantize-GEMM CUDA Kernel
__global__ void matmul_w4a16_kernel(
    const uint8_t* __restrict__ A_packed,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K,
    int group_size
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    int tid = threadIdx.x;
    int lane = tid % 32;

    float sum = 0.0f;
    int num_groups_per_row = K / group_size;
    const uint8_t* row_A = A_packed + (size_t)row * (K / 2);
    const float* row_scales = scales + (size_t)row * num_groups_per_row;
    const float* row_zeros = zeros + (size_t)row * num_groups_per_row;

    // Process elements in pairs (since 1 byte = two 4-bit weights)
    for (int k_byte = tid; k_byte < K / 2; k_byte += blockDim.x) {
        int k0 = k_byte * 2;
        int k1 = k0 + 1;

        uint8_t packed_val = row_A[k_byte];
        uint8_t w0_u4 = packed_val & 0x0F;
        uint8_t w1_u4 = (packed_val >> 4) & 0x0F;

        int g0 = k0 / group_size;
        int g1 = k1 / group_size;

        float s0 = row_scales[g0];
        float z0 = row_zeros[g0];
        float s1 = row_scales[g1];
        float z1 = row_zeros[g1];

        float w0_fp32 = (static_cast<float>(w0_u4) - z0) * s0;
        float w1_fp32 = (static_cast<float>(w1_u4) - z1) * s1;

        sum += w0_fp32 * x[k0] + w1_fp32 * x[k1];
    }

    // Warp-level reduction
    sum = warp_reduce_sum_q4(sum);

    static __shared__ float shared_sums[8][32];

    int warp_id = tid / 32;
    if (lane == 0) {
        shared_sums[threadIdx.y][warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (tid < (blockDim.x / 32)) ? shared_sums[threadIdx.y][tid] : 0.0f;
        block_sum = warp_reduce_sum_q4(block_sum);
        if (tid == 0) {
            y[row] = block_sum;
        }
    }
}

void matmul_w4a16(
    const uint8_t* A_packed_ptr,
    const float* scales_ptr,
    const float* zeros_ptr,
    const float* x_ptr,
    float* y_ptr,
    int M,
    int K,
    int group_size,
    cudaStream_t stream
) {
    dim3 block(32, 4);
    dim3 grid((M + block.y - 1) / block.y);

    matmul_w4a16_kernel<<<grid, block, 0, stream>>>(
        A_packed_ptr, scales_ptr, zeros_ptr, x_ptr, y_ptr, M, K, group_size
    );
}

} // namespace qwen

extern "C" {
    void run_matmul_w4a16(
        const uint8_t* A_packed,
        const float* scales,
        const float* zeros,
        const float* x,
        float* y,
        int M,
        int K,
        int group_size
    ) {
        qwen::matmul_w4a16(A_packed, scales, zeros, x, y, M, K, group_size, nullptr);
        cudaDeviceSynchronize();
    }
}
