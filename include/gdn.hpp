#ifndef QWEN_GDN_HPP
#define QWEN_GDN_HPP

#include <cuda_runtime.h>

namespace qwen {

    // --- GDN Convolution Step ---
    void gdn_conv_step(
        const float* input,
        const float* weight,
        float*       history,
        float*       output,
        int          channels,
        int          kernel,
        cudaStream_t stream
    );

    // --- Q/K Normalization ---
    void gdn_qk_normalize(
        float*       q,
        float*       k,
        int          heads,
        int          dim,
        float        eps,
        cudaStream_t stream
    );

    // --- Gating Calculations ---
    void gdn_gate_values(
        const float* beta_raw,
        const float* alpha_raw,
        const float* a,
        const float* dt_bias,
        float*       beta,
        float*       decay,
        int          heads,
        cudaStream_t stream
    );

    // --- Recurrent State Update ---
    void gdn_recurrent_step(
        float*       state,
        const float* q,
        const float* k,
        const float* v,
        const float* beta,
        const float* decay,
        float*       out,
        int          key_heads,
        int          value_heads,
        int          key_dim,
        int          value_dim,
        cudaStream_t stream
    );

    // --- Output Normalization and Gating ---
    void gdn_norm_gate(
        const float* x,
        const float* z,
        const float* weight,
        float*       out,
        int          heads,
        int          dim,
        float        eps,
        cudaStream_t stream
    );

} // namespace qwen

#endif // QWEN_GDN_HPP
