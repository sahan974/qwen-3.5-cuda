#include <iostream>
#include "config.hpp"
#include "cuda_utils.hpp"
#include "loader_gguf.hpp"
#include "model.hpp"

int main(int argc, char** argv) {
    std::cout << "=== qwen-3.5-cuda: Quantized Qwen 3.5 C++/CUDA Inference Engine ===" << std::endl;
    
    qwen::ModelConfig real_cfg = qwen::ModelConfig::real_config();
    std::cout << "\n--- Real Qwen3.5-35B-A3B Config ---" << std::endl;
    std::cout << "d_model: " << real_cfg.d_model << std::endl;
    std::cout << "n_layers: " << real_cfg.n_layers << std::endl;
    std::cout << "vocab_size: " << real_cfg.vocab_size << std::endl;

    qwen::ModelConfig test_cfg = qwen::ModelConfig::test_config();
    std::cout << "\n--- Test Config ---" << std::endl;
    std::cout << "d_model: " << test_cfg.d_model << ", n_layers: " << test_cfg.n_layers << std::endl;

    std::cout << "\n--- Testing CUDA Utils & CudaContext ---" << std::endl;
    try {
        qwen::CudaContext ctx;
        std::cout << "CudaContext initialized successfully (cublasHandle & cudaStream created)." << std::endl;

        std::cout << "\n--- Testing Full QwenModel Engine Initialization ---" << std::endl;
        qwen::GgufLoader loader;
        qwen::QwenModel model;
        if (model.init(test_cfg, loader)) {
            std::cout << "Running single decode step test..." << std::endl;
            int next_tok = model.decode_step(1, 0, ctx);
            std::cout << "Engine decode step succeeded! Predicted token ID: " << next_tok << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Engine test skipped/failed: " << e.what() << std::endl;
    }

    if (argc > 1) {
        std::cout << "\n--- Testing GGUF Tensor Loading to GPU ---" << std::endl;
        std::string gguf_path = argv[1];
        qwen::GgufLoader loader;
        if (loader.open(gguf_path)) {
            loader.print_summary();
            std::cout << "\nInitiating GPU transfer..." << std::endl;
            if (loader.load_tensors_to_gpu()) {
                std::cout << "All tensors safely residing in GPU VRAM!" << std::endl;
                loader.unload_gpu();
                std::cout << "VRAM freed successfully." << std::endl;
            }
        }
    }

    return 0;
}
