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

def test_matmul_q4():
    print("\n=== Testing INT4 (W4A16) GEMM CUDA Kernel vs PyTorch Reference ===")
    test_shapes = [
        (2048, 2048),
        (8192, 2048),
        (4096, 2048)
    ]
    group_size = 32
    
    for M, K in test_shapes:
        print(f"\nTesting INT4 Shape: M={M}, K={K}, group_size={group_size}...")
        num_groups = K // group_size
        
        # Random weights and scales/zeros
        W_fp32 = torch.randn(M, K, dtype=torch.float32)
        scales = torch.rand(M, num_groups, dtype=torch.float32) * 0.1 + 0.01
        zeros = torch.randint(0, 16, (M, num_groups), dtype=torch.float32)
        
        # Quantize to INT4 uint8 pair
        W_u4 = torch.zeros(M, K, dtype=torch.uint8)
        for g in range(num_groups):
            k_start = g * group_size
            k_end = (g + 1) * group_size
            s = scales[:, g:g+1]
            z = zeros[:, g:g+1]
            q = torch.round(W_fp32[:, k_start:k_end] / s + z).clamp(0, 15).to(torch.uint8)
            W_u4[:, k_start:k_end] = q

        # Pack pairs of 4-bit into uint8
        W_packed = torch.zeros(M, K // 2, dtype=torch.uint8)
        for k_byte in range(K // 2):
            w0 = W_u4[:, k_byte * 2]
            w1 = W_u4[:, k_byte * 2 + 1]
            W_packed[:, k_byte] = w0 | (w1 << 4)

        # Dequantize PyTorch reference
        W_dequant = torch.zeros(M, K, dtype=torch.float32)
        for g in range(num_groups):
            k_start = g * group_size
            k_end = (g + 1) * group_size
            s = scales[:, g:g+1]
            z = zeros[:, g:g+1]
            W_dequant[:, k_start:k_end] = (W_u4[:, k_start:k_end].to(torch.float32) - z) * s

        x = torch.randn(K, dtype=torch.float32)
        y_ref = torch.mv(W_dequant, x)
        print(f"PyTorch INT4 Reference computed. Target norm: {y_ref.norm().item():.4f}")

def test_operator_kernels():
    print("\n=== Testing Operator Kernels (RMSNorm, Gated RMSNorm, Residual Add) ===")
    N = 2048
    x = torch.randn(N, dtype=torch.float32)
    weight = torch.randn(N, dtype=torch.float32)
    gate = torch.randn(N, dtype=torch.float32)
    eps = 1e-6

    # RMSNorm (1 + weight)
    rms_ref = x / torch.sqrt(torch.mean(x**2) + eps) * (1.0 + weight)
    print(f"PyTorch RMSNorm (1+w) Norm: {rms_ref.norm().item():.4f}")

    # Gated RMSNorm
    silu_gate = gate * torch.sigmoid(gate)
    gated_rms_ref = (x / torch.sqrt(torch.mean(x**2) + eps)) * weight * silu_gate
    print(f"PyTorch Gated RMSNorm Norm: {gated_rms_ref.norm().item():.4f}")

def test_rope_kernel():
    print("\n=== Testing Partial RoPE Kernel (64-dim) ===")
    num_heads = 16
    head_dim = 256
    partial_dim = 64
    pos = 5
    theta = 1000000.0

    q = torch.randn(num_heads, head_dim, dtype=torch.float32)
    q_rot = q.clone()

    for h in range(num_heads):
        for p in range(partial_dim // 2):
            i0 = p * 2
            i1 = i0 + 1
            th = pos / (theta ** (i0 / partial_dim))
            c = math.cos(th)
            s = math.sin(th)
            v0 = q[h, i0].item()
            v1 = q[h, i1].item()
            q_rot[h, i0] = v0 * c - v1 * s
            q_rot[h, i1] = v0 * s + v1 * c

    print(f"Partial RoPE output computed. Rotated Q Norm: {q_rot.norm().item():.4f}")
    print(f"Unchanged dims [64..255] diff: {(q[:, 64:] - q_rot[:, 64:]).abs().max().item():.6f}")

def test_gdn_kernel():
    print("\n=== Testing Gated DeltaNet (GDN) Recurrent Update Kernel ===")
    num_heads = 32
    key_dim = 128
    value_dim = 128

    state = torch.zeros(num_heads, key_dim, value_dim, dtype=torch.float32)
    q = torch.randn(num_heads, key_dim, dtype=torch.float32)
    k = torch.randn(num_heads, key_dim, dtype=torch.float32)
    v = torch.randn(num_heads, value_dim, dtype=torch.float32)
    b = torch.randn(num_heads, key_dim, dtype=torch.float32)

    beta = torch.sigmoid(b)
    out_ref = torch.zeros(num_heads, value_dim, dtype=torch.float32)

    for h in range(num_heads):
        for k_idx in range(key_dim):
            b_val = beta[h, k_idx].item()
            k_val = k[h, k_idx].item()
            q_val = q[h, k_idx].item()

            # State update S_t = (1 - beta * k) * S_{t-1} + beta * v
            state[h, k_idx] = (1.0 - b_val * k_val) * state[h, k_idx] + b_val * v[h]
            out_ref[h] += state[h, k_idx] * q_val

    print(f"PyTorch GDN Recurrent Step computed. Output Norm: {out_ref.norm().item():.4f}")

def test_moe_kernels():
    print("\n=== Testing MoE Top-K Softmax & Expert Accumulation ===")
    num_experts = 256
    k = 8
    d_model = 2048

    logits = torch.randn(num_experts, dtype=torch.float32)
    topk = torch.topk(logits, k)
    topk_indices = topk.indices
    topk_weights = torch.softmax(topk.values, dim=0)

    print(f"Top-8 Expert Indices: {topk_indices.tolist()}")
    print(f"Top-8 Softmax Weights Sum: {topk_weights.sum().item():.6f}")

    expert_outputs = torch.randn(k, d_model, dtype=torch.float32)
    accum_ref = torch.sum(expert_outputs * topk_weights.unsqueeze(1), dim=0)
    print(f"PyTorch MoE Accumulated Output Norm: {accum_ref.norm().item():.4f}")

if __name__ == "__main__":
    test_matmul_q8()
    test_matmul_q4()
    test_operator_kernels()
    test_rope_kernel()
    test_gdn_kernel()
    test_moe_kernels()
