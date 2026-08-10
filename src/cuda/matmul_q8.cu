#include "matmul_q8.hpp"
#include <cuda_fp16.h>
#include <stdio.h>

namespace qwen {

// Warp reduction helper using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel: Each threadblock handles a group of rows (M), threads in block process K dimension in parallel
__global__ void matmul_w8a16_kernel(
    const int8_t* __restrict__ A,
    const float* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int K
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    float scale = scales[row];
    float sum = 0.0f;

    // Vectorized load or strided loop over K
    int tid = threadIdx.x;
    int lane = tid % 32;

    const int8_t* row_A = A + (size_t)row * K;

    for (int k = tid; k < K; k += blockDim.x) {
        float w_val = static_cast<float>(row_A[k]) * scale;
        float x_val = x[k];
        sum += w_val * x_val;
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // Shared memory for block reduction across warps if blockDim.x > 32
    static __shared__ float shared_sums[8][32]; // supports up to 256 threads (8 warps) per row group

    int warp_id = tid / 32;
    if (lane == 0) {
        shared_sums[threadIdx.y][warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces warp sums
    if (warp_id == 0) {
        float block_sum = (tid < (blockDim.x / 32)) ? shared_sums[threadIdx.y][tid] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (tid == 0) {
            y[row] = block_sum;
        }
    }
}

void matmul_w8a16(
    const int8_t* A_ptr,
    const float* scale_ptr,
    const float* x_ptr,
    float* y_ptr,
    int M,
    int K,
    cudaStream_t stream
) {
    // 32 threads per row for K reduction, 4 rows per block -> block size 128
    dim3 block(32, 4);
    dim3 grid((M + block.y - 1) / block.y);

    matmul_w8a16_kernel<<<grid, block, 0, stream>>>(A_ptr, scale_ptr, x_ptr, y_ptr, M, K);
}

} // namespace qwen

// C API wrapper for Python ctypes testing
extern "C" {
    void run_matmul_w8a16(
        const int8_t* A,
        const float* scales,
        const float* x,
        float* y,
        int M,
        int K
    ) {
        qwen::matmul_w8a16(A, scales, x, y, M, K, nullptr);
        cudaDeviceSynchronize();
    }
}
