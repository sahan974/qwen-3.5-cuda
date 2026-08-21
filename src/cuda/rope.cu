#include "rope.hpp"
#include "cuda_utils.hpp"
#include <cmath>
#include <stdexcept>

namespace qwen {

    // --- RoPE Kernel ---
    __global__ void rope_kernel(float* x, int pos, int heads, int hd, int rd, float base) {
        int h    = blockIdx.x;
        int i    = threadIdx.x;
        int half = rd / 2;

        if (h >= heads || i >= half) return;

        float* p = x + h * hd;

        // Calculate theta for the current dimension pair
        float theta = pos / powf(base, (2.0f * i) / rd);
        float c     = cosf(theta);
        float s     = sinf(theta);

        float a = p[i];
        float b = p[i + half];

        // Apply rotation
        p[i]        = a * c - b * s;
        p[i + half] = b * c + a * s;
    }

    // --- Host Wrapper ---
    void rope_forward(float* x, int pos, int heads, int hd, int rd, float base, cudaStream_t s) {
        if (!x ||
            pos < 0 ||
            heads <= 0 ||
            hd <= 0 ||
            rd <= 0 ||
            rd > hd ||
            (rd & 1) ||
            (rd / 2 > 1024) ||
            !(base > 0.0f)) {
            throw std::runtime_error("invalid RoPE arguments");
            }

        rope_kernel<<<heads, rd / 2, 0, s>>>(x, pos, heads, hd, rd, base);
        CUDA_CHECK(cudaGetLastError());
    }

} // namespace qwen
