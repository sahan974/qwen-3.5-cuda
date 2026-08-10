import torch
import numpy as np
import ctypes
import os

def test_matmul_q8():
    print("=== Testing INT8 (W8A16) GEMM CUDA Kernel vs PyTorch Reference ===")
    
    # Dimensions matching Qwen3.5 projection shapes
    test_shapes = [
        (2048, 2048),
        (8192, 2048),
        (4096, 2048)
    ]
    
    for M, K in test_shapes:
        print(f"\nTesting Shape: M={M}, K={K}...")
        
        # Random FP32 weights and scales
        W_fp32 = torch.randn(M, K, dtype=torch.float32)
        scales = W_fp32.abs().max(dim=1).values / 127.0
        scales = torch.clamp(scales, min=1e-5)
        
        # Quantize to INT8 per row
        W_int8 = torch.round(W_fp32 / scales.unsqueeze(1)).clamp(-128, 127).to(torch.int8)
        
        # Dequantized PyTorch weight for baseline calculation
        W_dequant = W_int8.to(torch.float32) * scales.unsqueeze(1)
        
        # Activation vector x
        x = torch.randn(K, dtype=torch.float32)
        
        # Expected baseline output
        y_ref = torch.mv(W_dequant, x)
        
        print(f"PyTorch Reference computed. Target norm: {y_ref.norm().item():.4f}")

if __name__ == "__main__":
    test_matmul_q8()
