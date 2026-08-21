#ifndef QWEN_CUDA_UTILS_HPP
#define QWEN_CUDA_UTILS_HPP

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace qwen {

// --- Error String Conversion ---
inline const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:           return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:   return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:      return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:     return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:     return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:     return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:  return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:    return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:     return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:     return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                              return "UNKNOWN_CUBLAS_ERROR";
    }
}

// --- Error Checking Macros ---
#define CUDA_CHECK(call)
    do {
        cudaError_t err = (call);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__
                      << " - " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
        }
    } while (0)

#define CUBLAS_CHECK(call)
    do {
        cublasStatus_t status = (call);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__
                      << " - " << qwen::cublasGetErrorString(status) << std::endl;
            throw std::runtime_error("cuBLAS Error: " + std::string(qwen::cublasGetErrorString(status)));
        }
    } while (0)

// --- CUDA Context Manager ---
class CudaContext {
public:
    // Lifecycle
    CudaContext() : handle_(nullptr), stream_(nullptr) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetStream(handle_, stream_));
    }

    ~CudaContext() {
        if (handle_) {
            cublasDestroy(handle_);
            handle_ = nullptr;
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    // Disable copying to prevent double-freeing resources
    CudaContext(const CudaContext&)            = delete;
    CudaContext& operator=(const CudaContext&) = delete;

    // Accessors
    cublasHandle_t handle() const { return handle_; }
    cudaStream_t   stream() const { return stream_; }

    // Synchronization
    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    cublasHandle_t handle_;
    cudaStream_t   stream_;
};

} // namespace qwen

#endif // QWEN_CUDA_UTILS_HPP
