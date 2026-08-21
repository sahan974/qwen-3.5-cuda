#include "matmul_q4.hpp"
#include "cuda_utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>

namespace qwen {
namespace {

// --- Inline Helpers ---

__device__ __forceinline__ float fp16_at(const uint8_t* p) {
    const uint16_t bits = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    return __half2float(*reinterpret_cast<const __half*>(&bits));
}

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// --- Dequantization Logic per GGML Type ---

__device__ __forceinline__ float dequant(const uint8_t* data, int type, uint64_t index) {
    // 1. F32
    if (type == 0) {
        return reinterpret_cast<const float*>(data)[index];
    }

    // 2. F16
    if (type == 1) {
        return __half2float(reinterpret_cast<const __half*>(data)[index]);
    }

    // 3. BF16
    if (type == 30) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(data)[index]);
    }

    // 4. Q8_0
    if (type == 8) {
        const uint8_t* b = data + (index / 32) * 34;
        return fp16_at(b) * static_cast<float>(static_cast<int8_t>(b[2 + index % 32]));
    }

    // 5. Q4_0
    if (type == 2) {
        const uint8_t* b = data + (index / 32) * 18;
        int i = index % 32;
        int q = (i < 16) ? (b[2 + i] & 15) : (b[2 + i - 16] >> 4);
        return fp16_at(b) * (q - 8);
    }

    // 6. Q5_0
    if (type == 6) {
        const uint8_t* b = data + (index / 32) * 22;
        int i = index % 32;
        uint32_t qh = static_cast<uint32_t>(b[2]) |
                      (static_cast<uint32_t>(b[3]) << 8) |
                      (static_cast<uint32_t>(b[4]) << 16) |
                      (static_cast<uint32_t>(b[5]) << 24);

        int j = i & 15;
        int high = (i < 16) ? ((qh >> j) & 1) : ((qh >> (j + 16)) & 1);
        int low  = (i < 16) ? (b[6 + j] & 15) : (b[6 + j] >> 4);
        return fp16_at(b) * ((low | (high << 4)) - 16);
    }

    // 7. Q4_K & Q5_K
    if (type == 12 || type == 13) {
        const int block_bytes = (type == 12) ? 144 : 176;
        const uint8_t* b = data + (index / 256) * block_bytes;
        const int i = index % 256;

        const float d    = fp16_at(b);
        const float dmin = fp16_at(b + 2);

        const uint8_t* scales = b + 4;
        const int sub = i / 32;
        uint8_t sc, mn;
        scale_min_k4(sub, scales, sc, mn);

        const int group64 = i / 64;
        const int in64    = i % 64;
        const int lane    = in64 & 31;

        if (type == 12) {
            const uint8_t q = b[16 + group64 * 32 + lane];
            return d * sc * ((in64 < 32) ? (q & 15) : (q >> 4)) - dmin * mn;
        }

        const uint8_t q = b[48 + group64 * 32 + lane];
        const uint8_t mask = static_cast<uint8_t>(1u << (2 * group64 + (in64 >= 32)));
        const int value = ((in64 < 32) ? (q & 15) : (q >> 4)) + ((b[16 + lane] & mask) ? 16 : 0);
        return d * sc * value - dmin * mn;
    }

    // 8. Q6_K
    if (type == 14) {
        const uint8_t* b = data + (index / 256) * 210;
        const int i = index % 256;

        const uint8_t* ql = b;
        const uint8_t* qh = b + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(b + 192);

        const int half = i / 128;
        const int r    = i % 128;
        const int lane = r & 31;
        const int quarter = r / 32;

        const uint8_t lo = (quarter == 0) ? (ql[half * 64 + lane] & 15) :
                           (quarter == 1) ? (ql[half * 64 + 32 + lane] & 15) :
                           (quarter == 2) ? (ql[half * 64 + lane] >> 4) :
                                            (ql[half * 64 + 32 + lane] >> 4);

        const uint8_t hi = (qh[half * 32 + lane] >> (2 * quarter)) & 3;
        const int scale_index = half * 8 + (lane / 16) + 2 * quarter;
        return fp16_at(b + 208) * scales[scale_index] * (static_cast<int>(lo | (hi << 4)) - 32);
    }

    return 0.0f;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

// --- Kernels ---

__global__ void gemv_kernel(const uint8_t* w, int type, const float* x, float* y,
                            int M, int K, uint64_t element_offset) {
    int row = blockIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    uint64_t base = element_offset + static_cast<uint64_t>(row) * K;

    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += dequant(w, type, base + k) * x[k];
    }

    sum = warp_sum(sum);

    __shared__ float partial[8];
    if ((threadIdx.x & 31) == 0) {
        partial[threadIdx.x >> 5] = sum;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        sum = (threadIdx.x < blockDim.x / 32) ? partial[threadIdx.x] : 0.0f;
        sum = warp_sum(sum);
        if (threadIdx.x == 0) {
            y[row] = sum;
        }
    }
}

__global__ void row_kernel(const uint8_t* w, int type, float* out, uint64_t start, int cols) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cols; i += blockDim.x * gridDim.x) {
        out[i] = dequant(w, type, start + i);
    }
}

} // anonymous namespace

// --- Host Interface ---

bool matmul_type_supported(GgmlType t) {
    return t == GgmlType::F32  || t == GgmlType::F16  || t == GgmlType::BF16 ||
           t == GgmlType::Q4_0 || t == GgmlType::Q5_0 || t == GgmlType::Q8_0 ||
           t == GgmlType::Q4_K || t == GgmlType::Q5_K || t == GgmlType::Q6_K;
}

void matmul_dispatch(const QuantTensor& w, const float* x, float* y, int M, int K,
                     cudaStream_t stream, uint64_t element_offset) {
    if (!w.device_ptr) {
        throw std::runtime_error("tensor '" + w.name + "' is not loaded");
    }
    if (!matmul_type_supported(w.type)) {
        throw std::runtime_error("unsupported GGML type for tensor '" + w.name + "'");
    }
    if (M <= 0 || K <= 0 || element_offset + static_cast<uint64_t>(M) * K > w.num_elements) {
        throw std::runtime_error("GEMV shape exceeds tensor '" + w.name + "'");
    }

    gemv_kernel<<<M, 256, 0, stream>>>(
        static_cast<const uint8_t*>(w.device_ptr),
        static_cast<int>(w.type),
        x,
        y,
        M,
        K,
        element_offset
    );
    CUDA_CHECK(cudaGetLastError());
}

void dequantize_row(const QuantTensor& t, float* out, int row, int cols, cudaStream_t stream) {
    if (!t.device_ptr || !matmul_type_supported(t.type)) {
        throw std::runtime_error("cannot dequantize tensor '" + t.name + "'");
    }

    uint64_t start = static_cast<uint64_t>(row) * cols;
    if (row < 0 || cols <= 0 || start + cols > t.num_elements) {
        throw std::runtime_error("row is outside tensor '" + t.name + "'");
    }

    row_kernel<<<(cols + 255) / 256, 256, 0, stream>>>(
        static_cast<const uint8_t*>(t.device_ptr),
        static_cast<int>(t.type),
        out,
        start,
        cols
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qwen
