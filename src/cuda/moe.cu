#include "cuda_utils.hpp"
#include <cmath>

namespace qwen {

// --- MoE Router Top-K Kernel ---
__global__ void topk_kernel(const float* x, int* idx, float* w, int n, int k) {
    if (threadIdx.x != 0) return;

    // Validate finite inputs
    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            idx[0] = -1;
            w[0]   = NAN;
            return;
        }
    }

    // Top-K Selection
    for (int j = 0; j < k; ++j) {
        float best = -INFINITY;
        int bi     = -1;

        for (int i = 0; i < n; ++i) {
            bool used = false;
            for (int p = 0; p < j; ++p) {
                used |= (idx[p] == i);
            }
            if (!used && x[i] > best) {
                best = x[i];
                bi   = i;
            }
        }
        idx[j] = bi;
        w[j]   = best;
    }

    // Softmax Normalization over Top-K
    float mx = w[0];
    for (int j = 1; j < k; ++j) {
        mx = fmaxf(mx, w[j]);
    }

    float sum = 0;
    for (int j = 0; j < k; ++j) {
        w[j] = expf(w[j] - mx);
        sum += w[j];
    }
    for (int j = 0; j < k; ++j) {
        w[j] /= sum;
    }
}

void moe_topk(const float* x, int* idx, float* w, int n, int k, cudaStream_t s) {
    topk_kernel<<<1, 1, 0, s>>>(x, idx, w, n, k);
    CUDA_CHECK(cudaGetLastError());
}


// --- SiLU Gate Multiplication Kernel ---
__global__ void silu_mul_kernel(const float* g, const float* u, float* h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float silu_g = g[i] / (1.0f + expf(-g[i]));
        h[i] = silu_g * u[i];
    }
}

void moe_silu_mul(const float* g, const float* u, float* h, int n, cudaStream_t s) {
    silu_mul_kernel<<<(n + 255) / 256, 256, 0, s>>>(g, u, h, n);
    CUDA_CHECK(cudaGetLastError());
}


// --- Expert Output Accumulation Kernel ---
__global__ void add_kernel(float* out, const float* x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] += a * x[i];
    }
}

void moe_add(float* out, const float* x, float a, int n, cudaStream_t s) {
    add_kernel<<<(n + 255) / 256, 256, 0, s>>>(out, x, a, n);
    CUDA_CHECK(cudaGetLastError());
}


// --- Shared Expert Output Kernel ---
__global__ void shared_kernel(float* out, const float* x, const float* gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sigmoid_gate = 1.0f / (1.0f + expf(-gate[0]));
        out[i] += x[i] * sigmoid_gate;
    }
}

void moe_shared_add(float* out, const float* x, const float* g, int n, cudaStream_t s) {
    shared_kernel<<<(n + 255) / 256, 256, 0, s>>>(out, x, g, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qwen
