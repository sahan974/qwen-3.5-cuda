#include "ops.hpp"
#include <cmath>
#include <stdio.h>

namespace qwen {

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum_ops(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// llama.cpp's Qwen3.5 GGUF converter adds 1 to ordinary RMSNorm weights.
// GGUF therefore stores the final multiplicative scale, not HF's zero-centered parameter.
__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int N,
    float eps
) {
    int tid = threadIdx.x;
    float sum_sq = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum_ops(sum_sq);

    __shared__ float shared_sum;
    int warp_id = tid / 32;
    static __shared__ float s_warps[32];

    if (tid % 32 == 0) {
        s_warps[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float b_sum = (tid < (blockDim.x / 32)) ? s_warps[tid] : 0.0f;
        b_sum = warp_reduce_sum_ops(b_sum);
        if (tid == 0) {
            shared_sum = rsqrtf(b_sum / static_cast<float>(N) + eps);
        }
    }
    __syncthreads();

    float rscale = shared_sum;
    for (int i = tid; i < N; i += blockDim.x) {
        out[i] = x[i] * rscale * weight[i];
    }
}

// Gated RMSNorm Kernel: norm(x) * weight * silu(gate)
__global__ void rmsnorm_gated_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gate,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int N,
    float eps
) {
    int tid = threadIdx.x;
    float sum_sq = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum_ops(sum_sq);

    __shared__ float shared_sum;
    int warp_id = tid / 32;
    static __shared__ float s_warps[32];

    if (tid % 32 == 0) {
        s_warps[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float b_sum = (tid < (blockDim.x / 32)) ? s_warps[tid] : 0.0f;
        b_sum = warp_reduce_sum_ops(b_sum);
        if (tid == 0) {
            shared_sum = rsqrtf(b_sum / static_cast<float>(N) + eps);
        }
    }
    __syncthreads();

    float rscale = shared_sum;
    for (int i = tid; i < N; i += blockDim.x) {
        out[i] = x[i] * rscale * weight[i] * silu(gate[i]);
    }
}

// Residual Add Kernel
__global__ void residual_add_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x1[idx] + x2[idx];
    }
}

// Embedding Gather Kernel
__global__ void encoder_kernel(
    int token_id,
    const float* __restrict__ embed_table,
    float* __restrict__ out,
    int d_model
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_model) {
        out[idx] = embed_table[(size_t)token_id * d_model + idx];
    }
}

void rmsnorm_forward(
    const float* x,
    const float* weight,
    float* out,
    int N,
    float eps,
    cudaStream_t stream
) {
    int threads = (N < 256) ? 128 : 256;
    rmsnorm_kernel<<<1, threads, 0, stream>>>(x, weight, out, N, eps);
}

void rmsnorm_gated_forward(
    const float* x,
    const float* gate,
    const float* weight,
    float* out,
    int N,
    float eps,
    cudaStream_t stream
) {
    int threads = (N < 256) ? 128 : 256;
    rmsnorm_gated_kernel<<<1, threads, 0, stream>>>(x, gate, weight, out, N, eps);
}

void residual_add(
    const float* x1,
    const float* x2,
    float* out,
    int size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(x1, x2, out, size);
}

void encoder_forward(
    int token_id,
    const float* embed_table,
    float* out,
    int d_model,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (d_model + threads - 1) / threads;
    encoder_kernel<<<blocks, threads, 0, stream>>>(token_id, embed_table, out, d_model);
}

} // namespace qwen
