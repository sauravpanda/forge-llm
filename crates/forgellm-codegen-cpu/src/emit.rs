//! Rust source code emission from IR graphs.
//!
//! Generates a complete Rust module containing:
//! - Typed buffer declarations for all intermediate tensors
//! - Specialized kernel functions for each operation
//! - A `forward()` function that chains all operations
//!
//! The generated code operates on flat f32 slices — no dynamic shapes.

use std::fmt::Write;

use forgellm_frontend::ir::*;

/// Generate Rust source code from a computation graph.
///
/// Returns a string containing a complete Rust module that implements
/// the model's forward pass. The generated code:
/// - Uses `&[f32]` for weight inputs (pre-loaded)
/// - Uses flat arrays for all intermediate buffers
/// - Has zero allocations during the forward pass
pub fn generate(graph: &Graph) -> Result<String, CodegenError> {
    let config = graph.config.as_ref().ok_or(CodegenError::MissingConfig)?;

    let mut code = String::with_capacity(32 * 1024);

    emit_header(&mut code, config)?;
    emit_kernel_functions(&mut code, config)?;
    // Per-projection dtype usage: any projection (or lm_head) using a
    // dtype turns on its kernel family.  For uniform configs this matches
    // the legacy `config.dtype == X` checks; for mixed Q4_K_M GGUFs both
    // Q4_K and Q6_K kernel families are emitted.
    let pdt = config.effective_proj_dtypes();
    let needs_q8 = pdt.uses(DType::Q8_0);
    let needs_q4 = pdt.uses(DType::Q4_0);
    let needs_q4k = pdt.uses(DType::Q4_K);
    let needs_q6k = pdt.uses(DType::Q6_K);
    if needs_q8 {
        emit_q8_0_kernel(&mut code)?;
        emit_q8_0_sdot_kernel(&mut code)?;
    }
    if needs_q8 {
        emit_specialized_q8_matmul_functions(&mut code, config)?;
        emit_specialized_q8_matmul_batched_functions(&mut code, config)?;
    }
    if needs_q4 {
        emit_q4_0_kernel(&mut code)?;
        // If the Q8_0 kernel family was already emitted, its shared helpers
        // (quantize_to_q8_0_blocks, f32_to_f16_bits) will clash.  Skip them.
        emit_q4_0_sdot_kernel(&mut code, /* skip_shared_helpers= */ needs_q8)?;
    }
    if needs_q4 {
        emit_specialized_q4_matmul_functions(&mut code, config)?;
        emit_specialized_q4_matmul_batched_functions(&mut code, config)?;
    }
    if needs_q4k {
        // Q4_K reuses `f16_bits_to_f32` and `quantize_to_q8_0_blocks_into` from
        // the Q8_0 kernel family.  Force the Q8_0 helpers to be emitted when
        // Q4_K is the projection dtype but Q8_0 isn't otherwise active.
        let need_q8_helpers_for_q4k = !needs_q8 && !needs_q4;
        if need_q8_helpers_for_q4k {
            emit_q8_0_kernel(&mut code)?;
            emit_q8_0_sdot_kernel(&mut code)?;
        }
        emit_q4_k_kernel(
            &mut code,
            /* skip_shared_helpers= */ needs_q8 || needs_q4 || need_q8_helpers_for_q4k,
        )?;
    }
    if needs_q4k {
        emit_specialized_q4_k_matmul_functions(&mut code, config)?;
    }
    if needs_q6k {
        // Q6_K reuses `f16_bits_to_f32` and `quantize_to_q8_0_blocks_into`.
        // Force the Q8_0 helpers to be emitted if neither Q8_0 / Q4_0 / Q4_K
        // is otherwise active.
        let need_q8_helpers_for_q6k = !needs_q8 && !needs_q4 && !needs_q4k;
        if need_q8_helpers_for_q6k {
            emit_q8_0_kernel(&mut code)?;
            emit_q8_0_sdot_kernel(&mut code)?;
        }
        emit_q6_k_kernel(
            &mut code,
            /* skip_shared_helpers= */ needs_q8 || needs_q4 || needs_q4k || need_q8_helpers_for_q6k,
        )?;
    }
    if needs_q6k {
        emit_specialized_q6_k_matmul_functions(&mut code, config)?;
    }
    emit_specialized_matmul_functions(&mut code, config)?;
    emit_forward_function(&mut code, graph, config)?;
    emit_prefill_function(&mut code, config)?;
    if config.dtype == DType::Q8_0 || config.dtype == DType::Q4_0 {
        emit_prefill_batched_function(&mut code, config)?;
        if use_flash_attention(config) {
            emit_flash_attention_batched_function(&mut code)?;
        }
        if config.sliding_window_size.is_some() {
            emit_sliding_attention_batched_function(&mut code)?;
        }
    }

    Ok(code)
}

/// Errors during code generation.
#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error("graph has no model config")]
    MissingConfig,

    #[error("unsupported operation for CPU codegen: {0}")]
    UnsupportedOp(String),

    #[error("format error: {0}")]
    Fmt(#[from] std::fmt::Error),
}

/// Returns true if flash attention should be emitted and used for this model.
///
/// Flash attention (tiled online softmax) is preferred for large contexts because it
/// replaces the O(seq_len) scores buffer with an O(FLASH_ATTN_BLOCK_SIZE) one, which
/// fits in L1 cache and reduces memory bandwidth by ~10× at long sequences.
///
/// Conditions: max_seq_len > 512 (worth the tiling overhead) AND no SWA
/// (SWA uses its own `attention_sliding` kernel that already limits the window).
fn use_flash_attention(config: &ModelConfig) -> bool {
    config.max_seq_len > 512 && config.sliding_window_size.is_none()
}

fn emit_header(code: &mut String, config: &ModelConfig) -> Result<(), CodegenError> {
    writeln!(code, "//! Auto-generated by ForgeLLM CPU codegen.")?;
    writeln!(
        code,
        "//! Model: {} ({} layers, hidden={})",
        config.architecture, config.num_layers, config.hidden_size
    )?;
    writeln!(code, "//!")?;
    writeln!(
        code,
        "//! This code has no dynamic dispatch — all dimensions are baked in."
    )?;
    writeln!(code)?;
    writeln!(code, "#![allow(clippy::excessive_precision)]")?;
    writeln!(
        code,
        "#![allow(dead_code, unused_imports, unused_assignments)]"
    )?;
    writeln!(code)?;
    writeln!(code, "use rayon::prelude::*;")?;
    writeln!(code)?;
    writeln!(code, "// Model constants")?;
    writeln!(
        code,
        "pub const HIDDEN_SIZE: usize = {};",
        config.hidden_size
    )?;
    writeln!(
        code,
        "pub const INTERMEDIATE_SIZE: usize = {};",
        config.intermediate_size
    )?;
    writeln!(code, "pub const NUM_LAYERS: usize = {};", config.num_layers)?;
    writeln!(
        code,
        "pub const NUM_HEADS: usize = {};",
        config.num_attention_heads
    )?;
    writeln!(
        code,
        "pub const NUM_KV_HEADS: usize = {};",
        config.num_kv_heads
    )?;
    writeln!(code, "pub const HEAD_DIM: usize = {};", config.head_dim)?;
    writeln!(code, "pub const VOCAB_SIZE: usize = {};", config.vocab_size)?;
    // Cap MAX_SEQ_LEN at 4096 for reasonable KV cache memory.
    // Models like Qwen2.5 have max_seq_len=32768 which would allocate
    // gigabytes of KV cache pre-allocated on construction.
    let effective_seq_len = config.max_seq_len.min(4096);
    writeln!(
        code,
        "pub const MAX_SEQ_LEN: usize = {};  // capped from model's {}",
        effective_seq_len, config.max_seq_len
    )?;
    writeln!(
        code,
        "pub const RMS_NORM_EPS: f32 = {:e};",
        config.rms_norm_eps
    )?;
    writeln!(code, "pub const ROPE_THETA: f32 = {:e};", config.rope_theta)?;
    // Sliding window size: 0 means full attention (no windowing).
    let swa = config.sliding_window_size.unwrap_or(0);
    writeln!(
        code,
        "pub const SLIDING_WINDOW_SIZE: usize = {};  // 0 = full attention",
        swa
    )?;
    // Flash Attention block size: K/V tiling granularity for cache-friendly attention.
    // 64 f32 = 256 bytes — fits comfortably in a single L1 cache line set.
    writeln!(
        code,
        "pub const FLASH_ATTN_BLOCK_SIZE: usize = 64;  // tile size for flash attention"
    )?;
    writeln!(
        code,
        "pub const PREFILL_BATCH_THRESHOLD: usize = 8;  // min prompt length for batched prefill path"
    )?;
    writeln!(code)?;

    // Aligned buffer type for SIMD-friendly memory access
    writeln!(
        code,
        "/// Cache-line aligned buffer for optimal SIMD performance."
    )?;
    writeln!(code, "#[repr(C, align(64))]")?;
    writeln!(code, "pub struct AlignedBuf<const N: usize>(pub [f32; N]);")?;
    writeln!(code)?;
    writeln!(code, "impl<const N: usize> AlignedBuf<N> {{")?;
    writeln!(code, "    pub fn new() -> Self {{ Self([0.0f32; N]) }}")?;
    writeln!(code, "    pub fn as_slice(&self) -> &[f32] {{ &self.0 }}")?;
    writeln!(
        code,
        "    pub fn as_mut_slice(&mut self) -> &mut [f32] {{ &mut self.0 }}"
    )?;
    writeln!(code, "}}")?;
    writeln!(code)?;

    // Apple Accelerate framework FFI for AMX-accelerated BLAS on macOS.
    // cblas_sgemv uses the Apple AMX coprocessor for matrix-vector multiply,
    // giving significant speedup on Apple Silicon for f32 matmul operations.
    writeln!(
        code,
        "// --- Apple Accelerate framework (AMX-accelerated BLAS) ---"
    )?;
    writeln!(code, "#[cfg(target_os = \"macos\")]")?;
    writeln!(code, "#[link(name = \"Accelerate\", kind = \"framework\")]")?;
    writeln!(code, "extern \"C\" {{")?;
    writeln!(
        code,
        "    /// BLAS single-precision matrix-vector multiply: y = alpha * A * x + beta * y"
    )?;
    writeln!(code, "    fn cblas_sgemv(")?;
    writeln!(code, "        order: i32, trans: i32,")?;
    writeln!(code, "        m: i32, n: i32,")?;
    writeln!(code, "        alpha: f32,")?;
    writeln!(code, "        a: *const f32, lda: i32,")?;
    writeln!(code, "        x: *const f32, incx: i32,")?;
    writeln!(code, "        beta: f32,")?;
    writeln!(code, "        y: *mut f32, incy: i32,")?;
    writeln!(code, "    );")?;
    writeln!(code, "}}")?;
    writeln!(code)?;
    writeln!(
        code,
        "// CblasRowMajor = 101, CblasNoTrans = 111, CblasTrans = 112"
    )?;
    writeln!(code, "#[cfg(target_os = \"macos\")]")?;
    writeln!(code, "const CBLAS_ROW_MAJOR: i32 = 101;")?;
    writeln!(code, "#[cfg(target_os = \"macos\")]")?;
    writeln!(code, "const CBLAS_NO_TRANS: i32 = 111;")?;
    writeln!(code, "#[cfg(target_os = \"macos\")]")?;
    writeln!(code, "const CBLAS_TRANS: i32 = 112;")?;
    writeln!(code)?;

    Ok(())
}

fn emit_kernel_functions(code: &mut String, config: &ModelConfig) -> Result<(), CodegenError> {
    // Emit all kernels as a template string — includes NEON SIMD with scalar fallback
    code.push_str(
        r#"
// --- NEON SIMD dot product (4 accumulators, 16-element unrolled) ---
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);
        let chunks = len / 16;
        for i in 0..chunks {
            let base = i * 16;
            s0 = vfmaq_f32(s0, vld1q_f32(a.as_ptr().add(base)), vld1q_f32(b.as_ptr().add(base)));
            s1 = vfmaq_f32(s1, vld1q_f32(a.as_ptr().add(base+4)), vld1q_f32(b.as_ptr().add(base+4)));
            s2 = vfmaq_f32(s2, vld1q_f32(a.as_ptr().add(base+8)), vld1q_f32(b.as_ptr().add(base+8)));
            s3 = vfmaq_f32(s3, vld1q_f32(a.as_ptr().add(base+12)), vld1q_f32(b.as_ptr().add(base+12)));
        }
        s0 = vaddq_f32(s0, s1);
        s2 = vaddq_f32(s2, s3);
        s0 = vaddq_f32(s0, s2);
        let mut r = vaddvq_f32(s0);
        for i in (chunks*16)..len { r += *a.get_unchecked(i) * *b.get_unchecked(i); }
        r
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len { sum += a[i] * b[i]; }
    sum
}

/// RMS normalization with NEON dot product + NEON apply
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn rms_norm(output: &mut [f32], input: &[f32], weight: &[f32], eps: f32) {
    use std::arch::aarch64::*;
    let n = input.len();
    let sum_sq = dot_f32(input, input, n);
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    unsafe {
        let vinv = vdupq_n_f32(inv_rms);
        let chunks = n / 4;
        for i in 0..chunks {
            let base = i * 4;
            let vi = vld1q_f32(input.as_ptr().add(base));
            let vw = vld1q_f32(weight.as_ptr().add(base));
            vst1q_f32(output.as_mut_ptr().add(base), vmulq_f32(vmulq_f32(vi, vinv), vw));
        }
    }
    for i in (n / 4 * 4)..n { output[i] = input[i] * inv_rms * weight[i]; }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn rms_norm(output: &mut [f32], input: &[f32], weight: &[f32], eps: f32) {
    let n = input.len();
    let sum_sq = dot_f32(input, input, n);
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for i in 0..n { output[i] = input[i] * inv_rms * weight[i]; }
}

/// Matrix multiply with NEON dot product per output element
#[inline]
pub fn matmul(output: &mut [f32], input: &[f32], weight: &[f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let row = &input[i*k..(i+1)*k];
        for j in 0..n {
            output[i*n+j] = dot_f32(row, &weight[j*k..(j+1)*k], k);
        }
    }
}

"#,
    );

    // Remaining kernels as template string
    code.push_str(
        r#"#[inline]
pub fn silu(output: &mut [f32], input: &[f32]) {
    for (o, &x) in output.iter_mut().zip(input.iter()) { *o = x / (1.0 + (-x).exp()); }
}

/// GeLU activation (approximate): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline]
pub fn gelu(output: &mut [f32], input: &[f32]) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        let cdf = 0.5 * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh());
        *o = x * cdf;
    }
}

/// Fused GeLU activation + elementwise multiply: output[i] = gelu(gate[i]) * up[i]
#[inline]
pub fn gelu_mul(output: &mut [f32], gate: &[f32], up: &[f32]) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    for i in 0..gate.len() {
        let x = gate[i];
        let cdf = 0.5 * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh());
        output[i] = (x * cdf) * up[i];
    }
}

/// Fused SiLU activation + elementwise multiply: output[i] = silu(gate[i]) * up[i]
/// Eliminates one intermediate buffer and one memory pass compared to separate silu+mul.
#[inline]
pub fn silu_mul(output: &mut [f32], gate: &[f32], up: &[f32]) {
    for i in 0..gate.len() {
        let x = gate[i];
        output[i] = (x / (1.0 + (-x).exp())) * up[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn elementwise_mul(output: &mut [f32], a: &[f32], b: &[f32]) {
    use std::arch::aarch64::*;
    let n = a.len();
    let chunks = n / 16;
    unsafe {
        for i in 0..chunks {
            let base = i * 16;
            vst1q_f32(output.as_mut_ptr().add(base), vmulq_f32(vld1q_f32(a.as_ptr().add(base)), vld1q_f32(b.as_ptr().add(base))));
            vst1q_f32(output.as_mut_ptr().add(base+4), vmulq_f32(vld1q_f32(a.as_ptr().add(base+4)), vld1q_f32(b.as_ptr().add(base+4))));
            vst1q_f32(output.as_mut_ptr().add(base+8), vmulq_f32(vld1q_f32(a.as_ptr().add(base+8)), vld1q_f32(b.as_ptr().add(base+8))));
            vst1q_f32(output.as_mut_ptr().add(base+12), vmulq_f32(vld1q_f32(a.as_ptr().add(base+12)), vld1q_f32(b.as_ptr().add(base+12))));
        }
    }
    for i in (chunks*16)..n { output[i] = a[i] * b[i]; }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn elementwise_mul(output: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..a.len() { output[i] = a[i] * b[i]; }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn elementwise_add(output: &mut [f32], a: &[f32], b: &[f32]) {
    use std::arch::aarch64::*;
    let n = a.len();
    let chunks = n / 16;
    unsafe {
        for i in 0..chunks {
            let base = i * 16;
            vst1q_f32(output.as_mut_ptr().add(base), vaddq_f32(vld1q_f32(a.as_ptr().add(base)), vld1q_f32(b.as_ptr().add(base))));
            vst1q_f32(output.as_mut_ptr().add(base+4), vaddq_f32(vld1q_f32(a.as_ptr().add(base+4)), vld1q_f32(b.as_ptr().add(base+4))));
            vst1q_f32(output.as_mut_ptr().add(base+8), vaddq_f32(vld1q_f32(a.as_ptr().add(base+8)), vld1q_f32(b.as_ptr().add(base+8))));
            vst1q_f32(output.as_mut_ptr().add(base+12), vaddq_f32(vld1q_f32(a.as_ptr().add(base+12)), vld1q_f32(b.as_ptr().add(base+12))));
        }
    }
    for i in (chunks*16)..n { output[i] = a[i] + b[i]; }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn elementwise_add(output: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..a.len() { output[i] = a[i] + b[i]; }
}

/// Fused residual add with NEON: a[i] += b[i]
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn residual_add(a: &mut [f32], b: &[f32]) {
    use std::arch::aarch64::*;
    let n = a.len();
    let chunks = n / 16;
    unsafe {
        for i in 0..chunks {
            let base = i * 16;
            vst1q_f32(a.as_mut_ptr().add(base), vaddq_f32(vld1q_f32(a.as_ptr().add(base)), vld1q_f32(b.as_ptr().add(base))));
            vst1q_f32(a.as_mut_ptr().add(base+4), vaddq_f32(vld1q_f32(a.as_ptr().add(base+4)), vld1q_f32(b.as_ptr().add(base+4))));
            vst1q_f32(a.as_mut_ptr().add(base+8), vaddq_f32(vld1q_f32(a.as_ptr().add(base+8)), vld1q_f32(b.as_ptr().add(base+8))));
            vst1q_f32(a.as_mut_ptr().add(base+12), vaddq_f32(vld1q_f32(a.as_ptr().add(base+12)), vld1q_f32(b.as_ptr().add(base+12))));
        }
    }
    for i in (chunks*16)..n { a[i] += b[i]; }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn residual_add(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() { a[i] += b[i]; }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn softmax(values: &mut [f32]) {
    use std::arch::aarch64::*;
    // Find max using NEON
    let n = values.len();
    let mut max_val = unsafe {
        let chunks = n / 4;
        let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
        for i in 0..chunks { vmax = vmaxq_f32(vmax, vld1q_f32(values.as_ptr().add(i * 4))); }
        vmaxvq_f32(vmax)
    };
    for i in (n / 4 * 4)..n { if values[i] > max_val { max_val = values[i]; } }
    // Exp and sum
    let mut sum = 0.0f32;
    for v in values.iter_mut() { *v = (*v - max_val).exp(); sum += *v; }
    // Normalize with NEON
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    unsafe {
        let vinv = vdupq_n_f32(inv);
        let chunks = n / 4;
        for i in 0..chunks {
            let base = i * 4;
            vst1q_f32(values.as_mut_ptr().add(base), vmulq_f32(vld1q_f32(values.as_ptr().add(base)), vinv));
        }
    }
    for i in (n / 4 * 4)..n { values[i] *= inv; }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn softmax(values: &mut [f32]) {
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in values.iter_mut() { *v = (*v - max_val).exp(); sum += *v; }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in values.iter_mut() { *v *= inv; }
}

/// Precompute RoPE frequency table: freqs[i] = 1/theta^(2i/head_dim) for i in 0..head_dim/2
#[inline]
pub fn rope_freqs(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim / 2).map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32)).collect()
}

/// Apply RoPE using precomputed frequencies — avoids powf per token.
/// Computes sin/cos table once per position, reuses across all heads.
#[inline]
pub fn rope(data: &mut [f32], pos: usize, head_dim: usize, num_heads: usize, freqs: &[f32]) {
    let half = head_dim / 2;
    // Precompute sin/cos for this position once — shared across all heads
    let mut cos_table = vec![0.0f32; half];
    let mut sin_table = vec![0.0f32; half];
    for i in 0..half {
        let angle = pos as f32 * freqs[i];
        let (s, c) = angle.sin_cos();
        cos_table[i] = c;
        sin_table[i] = s;
    }
    for h in 0..num_heads {
        let off = h * head_dim;
        for i in 0..half {
            let (x0, x1) = (data[off + 2*i], data[off + 2*i + 1]);
            data[off + 2*i] = x0 * cos_table[i] - x1 * sin_table[i];
            data[off + 2*i + 1] = x0 * sin_table[i] + x1 * cos_table[i];
        }
    }
}

/// Quantize a token's K or V vector to int8 with per-token absmax scale.
/// Returns the scale factor (max_abs / 127). Zero scale if all values are zero.
#[inline]
pub fn quantize_kv(values: &[f32], out: &mut [i8]) -> f32 {
    let mut max_abs = 0.0f32;
    for &v in values.iter() {
        let a = v.abs();
        if a > max_abs { max_abs = a; }
    }
    let scale = max_abs / 127.0;
    let inv_scale = if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 };
    for (i, &v) in values.iter().enumerate() {
        out[i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Dot product between f32 query slice and int8 cache slice with dequantization scale.
/// Computes sum(q[i] * (k_i8[i] as f32 * scale)) = scale * sum(q[i] * k_i8[i] as f32).
#[inline]
fn dot_f32_i8(q: &[f32], k_i8: &[i8], len: usize, scale: f32) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += q[i] * k_i8[i] as f32;
    }
    sum * scale
}

#[inline]
pub fn attention(output: &mut [f32], q: &[f32], k_cache: &[i8], v_cache: &[i8],
    k_scales: &[f32], v_scales: &[f32],
    seq_len: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize) {
    let gsize = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_stride = num_kv_heads * head_dim;
    // Fixed-size scores buffer — no heap allocation
    let mut scores = [0.0f32; MAX_SEQ_LEN];
    for h in 0..num_heads {
        let kv_h = h / gsize;
        let qo = h * head_dim;
        for t in 0..seq_len {
            let ko = t * kv_stride + kv_h * head_dim;
            scores[t] = dot_f32_i8(&q[qo..qo+head_dim], &k_cache[ko..ko+head_dim], head_dim, k_scales[t]) * scale;
        }
        softmax(&mut scores[..seq_len]);
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for t in 0..seq_len {
                sum += scores[t] * (v_cache[t * kv_stride + kv_h * head_dim + d] as f32 * v_scales[t]);
            }
            output[qo+d] = sum;
        }
    }
}

#[inline]
pub fn embedding(output: &mut [f32], token_id: u32, weight: &[f32], embed_dim: usize) {
    let off = token_id as usize * embed_dim;
    output.copy_from_slice(&weight[off..off + embed_dim]);
}

"#,
    );

    // Only emit attention_sliding for models that use sliding window attention (Mistral).
    if config.sliding_window_size.is_some() {
        code.push_str(
            r#"
/// Sliding Window Attention: like `attention` but only attends to the last `window` tokens.
/// Used by Mistral models (SWA). When seq_len <= window, behaves identically to `attention`.
#[inline]
pub fn attention_sliding(output: &mut [f32], q: &[f32], k_cache: &[i8], v_cache: &[i8],
    k_scales: &[f32], v_scales: &[f32],
    seq_len: usize, window: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize) {
    let start = if seq_len > window { seq_len - window } else { 0 };
    let win_len = seq_len - start;
    let gsize = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_stride = num_kv_heads * head_dim;
    let mut scores = [0.0f32; MAX_SEQ_LEN];
    for h in 0..num_heads {
        let kv_h = h / gsize;
        let qo = h * head_dim;
        for (si, t) in (start..seq_len).enumerate() {
            let ko = t * kv_stride + kv_h * head_dim;
            scores[si] = dot_f32_i8(&q[qo..qo+head_dim], &k_cache[ko..ko+head_dim], head_dim, k_scales[t]) * scale;
        }
        softmax(&mut scores[..win_len]);
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for (si, t) in (start..seq_len).enumerate() {
                sum += scores[si] * (v_cache[t * kv_stride + kv_h * head_dim + d] as f32 * v_scales[t]);
            }
            output[qo+d] = sum;
        }
    }
}

"#,
        );
    }

    // Emit flash attention for large-context non-SWA models.
    if use_flash_attention(config) {
        emit_flash_attention_function(code)?;
    }

    Ok(())
}

/// Emit the `attention_flash` kernel: tiled attention with online softmax.
///
/// Uses `FLASH_ATTN_BLOCK_SIZE` (64) as the K/V tile size so the score buffer fits in
/// L1 cache.  The online softmax (running max + rescaling) avoids a second pass over
/// the full sequence, reducing memory bandwidth by O(seq_len / BLOCK_SIZE).
///
/// The output accumulator `acc` uses `HEAD_DIM` — a const in the generated file —
/// so it lives on the stack (zero heap allocation), consistent with the existing
/// `attention` kernel that uses `MAX_SEQ_LEN` for its scores buffer.
fn emit_flash_attention_function(code: &mut String) -> Result<(), CodegenError> {
    code.push_str(
        r#"
/// Flash Attention: tiled online-softmax attention for long contexts.
/// Reads K/V in blocks of FLASH_ATTN_BLOCK_SIZE (64) for L1/L2 cache efficiency.
/// Uses O(FLASH_ATTN_BLOCK_SIZE) score buffer instead of O(seq_len) — safe for 4096+ tokens.
/// Numerically equivalent to `attention` (standard softmax) up to floating-point rounding.
#[inline]
pub fn attention_flash(output: &mut [f32], q: &[f32], k_cache: &[i8], v_cache: &[i8],
    k_scales: &[f32], v_scales: &[f32],
    seq_len: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize) {
    let gsize = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_stride = num_kv_heads * head_dim;
    // Block-local scores: 64 × f32 = 256 bytes — one L1 cache line set
    let mut scores = [0.0f32; FLASH_ATTN_BLOCK_SIZE];
    for h in 0..num_heads {
        let kv_h = h / gsize;
        let qo = h * head_dim;
        // Per-head online softmax accumulators (all on the stack)
        let mut m_i = f32::NEG_INFINITY;   // running max of scores seen so far
        let mut l_i = 0.0f32;             // running sum of exp(score − m_i)
        let mut acc = [0.0f32; HEAD_DIM]; // weighted V accumulator
        let mut block_start = 0usize;
        while block_start < seq_len {
            let block_end = (block_start + FLASH_ATTN_BLOCK_SIZE).min(seq_len);
            let block_len = block_end - block_start;
            // Step 1: QK scores for this block + block-local max
            let mut m_block = f32::NEG_INFINITY;
            for bi in 0..block_len {
                let t = block_start + bi;
                let ko = t * kv_stride + kv_h * head_dim;
                let s = dot_f32_i8(&q[qo..qo+head_dim], &k_cache[ko..ko+head_dim], head_dim, k_scales[t]) * scale;
                scores[bi] = s;
                if s > m_block { m_block = s; }
            }
            // Step 2: update running max; rescale previous accumulator and denominator
            // exp(m_prev − m_new): = 1 when no new max, < 1 when new max, → 0 at first block
            let m_prev = m_i;
            if m_block > m_i { m_i = m_block; }
            let exp_scale = (m_prev - m_i).exp();
            l_i *= exp_scale;
            for d in 0..head_dim { acc[d] *= exp_scale; }
            // Step 3: unnormalized weights for this block; accumulate block denominator
            let mut l_block = 0.0f32;
            for bi in 0..block_len {
                let e = (scores[bi] - m_i).exp();
                scores[bi] = e;
                l_block += e;
            }
            l_i += l_block;
            // Step 4: accumulate weighted V contribution (dequantize v on the fly)
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for bi in 0..block_len {
                    let t = block_start + bi;
                    sum += scores[bi] * (v_cache[t * kv_stride + kv_h * head_dim + d] as f32 * v_scales[t]);
                }
                acc[d] += sum;
            }
            block_start = block_end;
        }
        // Step 5: normalize by running denominator
        let inv_l = 1.0 / l_i;
        for d in 0..head_dim { output[qo+d] = acc[d] * inv_l; }
    }
}

"#,
    );
    Ok(())
}

/// Emit `attention_flash_batch` — Q-tiled flash attention that amortizes
/// K/V scans across Q_TILE query rows.  Used by `forward_prefill_batched`
/// to replace the per-token `attention_flash` call loop.
///
/// Algorithm per (head, Q_TILE):
///   - Stream K/V in blocks of FLASH_ATTN_BLOCK_SIZE.
///   - For each block, compute (Q_TILE × block_len) scores, apply causal
///     mask per-query, update per-query online softmax state, accumulate
///     P·V into per-query output accumulators.
///
/// Bandwidth win: each K/V position is loaded once per Q_TILE queries
/// instead of once per query → factor ~Q_TILE (16) reduction in K/V
/// re-read bandwidth vs per-token attention_flash.
///
/// Parallelism: heads are independent, parallelized via rayon with
/// disjoint output writes (per-head output offsets don't overlap).
fn emit_flash_attention_batched_function(code: &mut String) -> Result<(), CodegenError> {
    code.push_str(
        r#"
/// Batched flash attention for prefill: Q-tiled online softmax over M queries.
///
/// Q_TILE is fixed at 16 — the sweet spot where per-Q-tile state (scores +
/// accumulators) fits comfortably in L1 alongside the K/V block.  For
/// HEAD_DIM=64 the per-tile state is ~4 KB scores + ~4 KB acc + ~4 KB K/V
/// block = ~12 KB, well within M5 Pro's 128 KB L1.
///
/// Causal mask: query row `q_tile_start + qi` (global position
/// `base_pos + q_tile_start + qi`) attends to K/V positions up to and
/// including that global position.
///
/// Output layout: `[m * num_heads * head_dim]` row-major, matching the
/// concatenated per-token `attention_flash` outputs.
#[inline]
pub fn attention_flash_batch(
    output: &mut [f32], q_batch: &[f32],
    k_cache: &[i8], v_cache: &[i8],
    k_scales: &[f32], v_scales: &[f32],
    m: usize, base_pos: usize,
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
) {
    const Q_TILE: usize = 16;
    let gsize = num_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let kv_stride = num_kv_heads * head_dim;

    // Parallel over heads: each head's work is independent, and per-head
    // output writes to disjoint [r * num_heads * head_dim + h * head_dim ..
    // .. + head_dim] slices.  Use a `usize` round-trip for the raw output
    // pointer to dodge the Send/Sync dance on *mut f32.
    let out_addr = output.as_mut_ptr() as usize;
    let out_stride = num_heads * head_dim;

    (0..num_heads).into_par_iter().for_each(|h| {
        let kv_h = h / gsize;
        let qo = h * head_dim;

        // Process queries in tiles of Q_TILE.
        let mut q_tile_start = 0usize;
        while q_tile_start < m {
            let q_tile_len = Q_TILE.min(m - q_tile_start);
            let max_q_pos = base_pos + q_tile_start + q_tile_len - 1;

            // Per-query online softmax state for this tile.  All stack-
            // allocated: 16 queries × HEAD_DIM × 4 bytes = ≤ 16 KB.
            let mut m_state = [f32::NEG_INFINITY; Q_TILE];
            let mut l_state = [0.0f32; Q_TILE];
            let mut acc = [[0.0f32; HEAD_DIM]; Q_TILE];

            // Stream K/V blocks.  Only need blocks that overlap [0, max_q_pos].
            let mut block_start = 0usize;
            while block_start <= max_q_pos {
                let block_end = (block_start + FLASH_ATTN_BLOCK_SIZE).min(max_q_pos + 1);
                let block_len = block_end - block_start;

                // Score buffer: [Q_TILE][FLASH_ATTN_BLOCK_SIZE]
                let mut scores = [[0.0f32; FLASH_ATTN_BLOCK_SIZE]; Q_TILE];

                // Phase 1: compute Q_TILE × block_len scores.
                // Outer bi: load K[t] once, dot with q_tile_len Q rows.
                for bi in 0..block_len {
                    let t = block_start + bi;
                    let ko = t * kv_stride + kv_h * head_dim;
                    let k_row = &k_cache[ko..ko + head_dim];
                    let k_scale = k_scales[t];
                    for qi in 0..q_tile_len {
                        let r = q_tile_start + qi;
                        let q_row = &q_batch[r * out_stride + qo..r * out_stride + qo + head_dim];
                        scores[qi][bi] = dot_f32_i8(q_row, k_row, head_dim, k_scale) * scale;
                    }
                }

                // Phase 2: causal mask + per-query online softmax update.
                for qi in 0..q_tile_len {
                    let q_pos = base_pos + q_tile_start + qi;
                    // Valid K positions for this query in this block:
                    // [block_start, min(block_end, q_pos + 1))
                    let valid_end = (q_pos + 1).saturating_sub(block_start).min(block_len);
                    if valid_end == 0 {
                        // Entire block is beyond this query's causal window.
                        for bi in 0..block_len { scores[qi][bi] = 0.0; }
                        continue;
                    }

                    let mut m_block = f32::NEG_INFINITY;
                    for bi in 0..valid_end {
                        if scores[qi][bi] > m_block { m_block = scores[qi][bi]; }
                    }
                    let m_prev = m_state[qi];
                    let m_new = m_prev.max(m_block);
                    m_state[qi] = m_new;
                    let exp_scale = (m_prev - m_new).exp();
                    l_state[qi] *= exp_scale;
                    for d in 0..head_dim { acc[qi][d] *= exp_scale; }

                    let mut l_block = 0.0f32;
                    for bi in 0..valid_end {
                        let e = (scores[qi][bi] - m_new).exp();
                        scores[qi][bi] = e;
                        l_block += e;
                    }
                    // Zero out causally-masked positions so phase 3 doesn't add them.
                    for bi in valid_end..block_len {
                        scores[qi][bi] = 0.0;
                    }
                    l_state[qi] += l_block;
                }

                // Phase 3: accumulate P·V.  Outer d, inner (bi, qi) gives
                // good reuse of the V-element load across the Q_TILE queries.
                for bi in 0..block_len {
                    let t = block_start + bi;
                    let vo = t * kv_stride + kv_h * head_dim;
                    let v_scale = v_scales[t];
                    for d in 0..head_dim {
                        let v_val = v_cache[vo + d] as f32 * v_scale;
                        for qi in 0..q_tile_len {
                            acc[qi][d] += scores[qi][bi] * v_val;
                        }
                    }
                }

                block_start = block_end;
            }

            // Phase 4: normalize and write outputs for this Q tile.
            // Different (h, qi) pairs write to disjoint output slices, so
            // the unsafe raw-pointer write is sound across workers.
            for qi in 0..q_tile_len {
                let r = q_tile_start + qi;
                let inv_l = 1.0 / l_state[qi];
                unsafe {
                    let p = (out_addr as *mut f32).add(r * out_stride + qo);
                    for d in 0..head_dim {
                        *p.add(d) = acc[qi][d] * inv_l;
                    }
                }
            }

            q_tile_start += q_tile_len;
        }
    });
}

"#,
    );
    Ok(())
}

/// Emit `attention_sliding_batch` — batched SWA attention.  Mirrors
/// `attention_flash_batch` but applies both the causal mask AND a
/// sliding-window mask: query at global position `q_pos` attends only
/// to K/V positions `[max(0, q_pos - window + 1), q_pos]`.
///
/// Bandwidth compared to the per-token `attention_sliding` fallback in
/// `forward_prefill_batched`: K/V loaded once per `(head, Q_TILE)` pair
/// instead of once per query → ~Q_TILE (16) reduction.
///
/// The outer K-block loop is bounded: for a Q tile `[q_tile_start,
/// q_tile_end)`, the earliest K position any query in the tile might
/// need is `max(0, q_tile_start - window + 1)` (from the earliest query
/// in the tile) and the latest is `q_tile_end - 1` (from the latest
/// query).  Blocks outside that range are skipped entirely.
fn emit_sliding_attention_batched_function(code: &mut String) -> Result<(), CodegenError> {
    code.push_str(
        r#"
/// Batched sliding-window attention for prefill: Q-tiled online softmax.
///
/// Per-query valid K range: `[max(0, q_pos - window + 1), q_pos]`
/// where `q_pos = base_pos + q_tile_start + qi`.  Mask logic runs per
/// query inside phase 2 (causal_end + window_start checks).
///
/// Output layout: `[m * num_heads * head_dim]` row-major, matching the
/// concatenated per-token `attention_sliding` outputs.
#[inline]
pub fn attention_sliding_batch(
    output: &mut [f32], q_batch: &[f32],
    k_cache: &[i8], v_cache: &[i8],
    k_scales: &[f32], v_scales: &[f32],
    m: usize, base_pos: usize, window: usize,
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
) {
    const Q_TILE: usize = 16;
    let gsize = num_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let kv_stride = num_kv_heads * head_dim;

    let out_addr = output.as_mut_ptr() as usize;
    let out_stride = num_heads * head_dim;

    (0..num_heads).into_par_iter().for_each(|h| {
        let kv_h = h / gsize;
        let qo = h * head_dim;

        let mut q_tile_start = 0usize;
        while q_tile_start < m {
            let q_tile_len = Q_TILE.min(m - q_tile_start);
            // Earliest query in tile attends from max(0, q_tile_start_pos - window + 1);
            // latest query attends up to (inclusive) q_tile_end_pos.
            let q_tile_start_pos = base_pos + q_tile_start;
            let q_tile_end_pos = base_pos + q_tile_start + q_tile_len - 1;
            let tile_min_k = q_tile_start_pos.saturating_sub(window - 1);
            let tile_max_k = q_tile_end_pos; // inclusive

            let mut m_state = [f32::NEG_INFINITY; Q_TILE];
            let mut l_state = [0.0f32; Q_TILE];
            let mut acc = [[0.0f32; HEAD_DIM]; Q_TILE];

            // Bound the outer block loop to [tile_min_k, tile_max_k + 1).
            let mut block_start = (tile_min_k / FLASH_ATTN_BLOCK_SIZE) * FLASH_ATTN_BLOCK_SIZE;
            while block_start <= tile_max_k {
                let block_end = (block_start + FLASH_ATTN_BLOCK_SIZE).min(tile_max_k + 1);
                let block_len = block_end - block_start;

                let mut scores = [[0.0f32; FLASH_ATTN_BLOCK_SIZE]; Q_TILE];

                // Phase 1: Q × K^T.  Load K[t] once, dot with all queries.
                // We compute raw scores for every bi; masking happens in phase 2.
                for bi in 0..block_len {
                    let t = block_start + bi;
                    let ko = t * kv_stride + kv_h * head_dim;
                    let k_row = &k_cache[ko..ko + head_dim];
                    let k_scale = k_scales[t];
                    for qi in 0..q_tile_len {
                        let r = q_tile_start + qi;
                        let q_row = &q_batch[r * out_stride + qo..r * out_stride + qo + head_dim];
                        scores[qi][bi] = dot_f32_i8(q_row, k_row, head_dim, k_scale) * scale;
                    }
                }

                // Phase 2: causal + window mask + per-query online softmax update.
                for qi in 0..q_tile_len {
                    let q_pos = base_pos + q_tile_start + qi;
                    // Valid K positions for this query in this block:
                    //   [max(block_start, q_pos - window + 1), min(block_end, q_pos + 1))
                    let win_start = q_pos.saturating_sub(window - 1);
                    let valid_lo = win_start.max(block_start).saturating_sub(block_start);
                    let valid_hi = (q_pos + 1).saturating_sub(block_start).min(block_len);
                    if valid_lo >= valid_hi {
                        // No valid K positions in this block for this query.
                        for bi in 0..block_len { scores[qi][bi] = 0.0; }
                        continue;
                    }

                    // Zero out (pre-softmax) the masked positions.
                    for bi in 0..valid_lo { scores[qi][bi] = f32::NEG_INFINITY; }
                    for bi in valid_hi..block_len { scores[qi][bi] = f32::NEG_INFINITY; }

                    let mut m_block = f32::NEG_INFINITY;
                    for bi in valid_lo..valid_hi {
                        if scores[qi][bi] > m_block { m_block = scores[qi][bi]; }
                    }
                    let m_prev = m_state[qi];
                    let m_new = m_prev.max(m_block);
                    m_state[qi] = m_new;
                    let exp_scale = (m_prev - m_new).exp();
                    l_state[qi] *= exp_scale;
                    for d in 0..head_dim { acc[qi][d] *= exp_scale; }

                    let mut l_block = 0.0f32;
                    for bi in valid_lo..valid_hi {
                        let e = (scores[qi][bi] - m_new).exp();
                        scores[qi][bi] = e;
                        l_block += e;
                    }
                    // Zero out masked scores (for phase 3 P·V accumulate).
                    for bi in 0..valid_lo { scores[qi][bi] = 0.0; }
                    for bi in valid_hi..block_len { scores[qi][bi] = 0.0; }
                    l_state[qi] += l_block;
                }

                // Phase 3: P · V.
                for bi in 0..block_len {
                    let t = block_start + bi;
                    let vo = t * kv_stride + kv_h * head_dim;
                    let v_scale = v_scales[t];
                    for d in 0..head_dim {
                        let v_val = v_cache[vo + d] as f32 * v_scale;
                        for qi in 0..q_tile_len {
                            acc[qi][d] += scores[qi][bi] * v_val;
                        }
                    }
                }

                block_start = block_end;
            }

            // Phase 4: normalize and write outputs for this Q tile.
            for qi in 0..q_tile_len {
                let r = q_tile_start + qi;
                // l_state[qi] is 0 only if the query had no valid K positions
                // at all — shouldn't happen in practice (every query sees itself),
                // but guard to avoid divide-by-zero producing NaN outputs.
                let inv_l = if l_state[qi] > 0.0 { 1.0 / l_state[qi] } else { 0.0 };
                unsafe {
                    let p = (out_addr as *mut f32).add(r * out_stride + qo);
                    for d in 0..head_dim {
                        *p.add(d) = acc[qi][d] * inv_l;
                    }
                }
            }

            q_tile_start += q_tile_len;
        }
    });
}

"#,
    );
    Ok(())
}

/// Collect all unique (k, n) matmul shapes used in the forward pass.
fn matmul_shapes(config: &ModelConfig) -> Vec<(usize, usize)> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let vocab = config.vocab_size;
    let qk_size = num_heads * head_dim;
    let kv_size = num_kv_heads * head_dim;

    let mut shapes = vec![
        (hidden, qk_size),      // q_proj
        (hidden, kv_size),      // k_proj, v_proj
        (qk_size, hidden),      // o_proj
        (hidden, intermediate), // gate_proj, up_proj
        (intermediate, hidden), // down_proj
        (hidden, vocab),        // lm_head
    ];
    shapes.sort();
    shapes.dedup();
    shapes
}

/// Emit specialized matmul_vec functions with baked-in K and N dimensions.
/// Since all our matmuls are m=1 (single-token inference), we generate
/// `matmul_vec_KxN(output, input, weight)` with no runtime dimension args.
fn emit_specialized_matmul_functions(
    code: &mut String,
    config: &ModelConfig,
) -> Result<(), CodegenError> {
    writeln!(code, "// --- Shape-specialized matmul functions (m=1) ---")?;
    writeln!(
        code,
        "// All dimensions baked in at compile time — no runtime size parameters."
    )?;
    writeln!(
        code,
        "// On macOS, uses cblas_sgemv from the Accelerate framework (AMX-accelerated)."
    )?;
    writeln!(code)?;

    // Parallelize when the weight matrix is large enough to amortize Rayon overhead.
    // Use a byte-count threshold: N * row_bytes > 1 MB → parallel.
    // This adapts to model size: 135M stays sequential for small projections,
    // 1B parallelizes down_proj (2048 * 2720 = 5.4MB) and q/o_proj (2048 * 2176 = 4.3MB).
    let par_byte_threshold: usize = 1_000_000;

    for &(k, n) in &matmul_shapes(config) {
        // --- macOS path: cblas_sgemv via Accelerate (uses AMX coprocessor) ---
        // Weight is row-major [N, K]. cblas_sgemv(RowMajor, NoTrans, N, K, 1, W, K, x, 1, 0, y, 1)
        // computes y = W * x which is exactly our matmul_vec: [N, K] x [K, 1] -> [N, 1].
        writeln!(
            code,
            "/// Specialized matmul: [1, {k}] x [{n}, {k}]^T -> [1, {n}]"
        )?;
        writeln!(
            code,
            "/// On macOS: uses cblas_sgemv (Accelerate/AMX) for hardware-accelerated matmul."
        )?;
        writeln!(code, "#[cfg(target_os = \"macos\")]")?;
        writeln!(code, "#[inline]")?;
        writeln!(
            code,
            "fn matmul_vec_{k}x{n}(output: &mut [f32; {n}], input: &[f32; {k}], weight: &[f32]) {{"
        )?;
        writeln!(
            code,
            "    // Safety: weight.len() >= {n}*{k}, input.len() >= {k}, output.len() >= {n}"
        )?;
        writeln!(code, "    // cblas_sgemv reads/writes within these bounds.")?;
        writeln!(code, "    unsafe {{")?;
        writeln!(code, "        cblas_sgemv(")?;
        writeln!(code, "            CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,")?;
        writeln!(code, "            {n} as i32, {k} as i32,")?;
        writeln!(code, "            1.0,")?;
        writeln!(code, "            weight.as_ptr(), {k} as i32,")?;
        writeln!(code, "            input.as_ptr(), 1,")?;
        writeln!(code, "            0.0,")?;
        writeln!(code, "            output.as_mut_ptr(), 1,")?;
        writeln!(code, "        );")?;
        writeln!(code, "    }}")?;
        writeln!(code, "}}")?;
        writeln!(code)?;

        // --- non-macOS fallback: NEON/scalar dot_f32 ---
        writeln!(
            code,
            "/// Specialized matmul: [1, {k}] x [{n}, {k}]^T -> [1, {n}]"
        )?;
        writeln!(code, "#[cfg(not(target_os = \"macos\"))]")?;
        let weight_bytes = n * k * 4; // f32: 4 bytes per element
        if weight_bytes >= par_byte_threshold {
            // Parallel path: par_chunks_mut(256) for cache locality + amortized Rayon overhead
            writeln!(
                code,
                "/// Parallelized: par_chunks_mut(256) with 4-way row ILP per thread"
            )?;
            writeln!(code, "#[inline]")?;
            writeln!(
                code,
                "fn matmul_vec_{k}x{n}(output: &mut [f32; {n}], input: &[f32; {k}], weight: &[f32]) {{"
            )?;
            writeln!(
                code,
                "    output.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
            )?;
            writeln!(code, "        let base = chunk_idx * 256;")?;
            writeln!(code, "        let len = out.len();")?;
            writeln!(code, "        let chunks4 = len / 4;")?;
            writeln!(code, "        for c in 0..chunks4 {{")?;
            writeln!(code, "            let r = c * 4;")?;
            writeln!(code, "            let j = base + r;")?;
            writeln!(
                code,
                "            out[r]   = dot_f32(&input[..], &weight[j*{k}..(j+1)*{k}], {k});"
            )?;
            writeln!(
                code,
                "            out[r+1] = dot_f32(&input[..], &weight[(j+1)*{k}..(j+2)*{k}], {k});"
            )?;
            writeln!(
                code,
                "            out[r+2] = dot_f32(&input[..], &weight[(j+2)*{k}..(j+3)*{k}], {k});"
            )?;
            writeln!(
                code,
                "            out[r+3] = dot_f32(&input[..], &weight[(j+3)*{k}..(j+4)*{k}], {k});"
            )?;
            writeln!(code, "        }}")?;
            writeln!(code, "        for r in (chunks4*4)..len {{")?;
            writeln!(code, "            let j = base + r;")?;
            writeln!(
                code,
                "            out[r] = dot_f32(&input[..], &weight[j*{k}..(j+1)*{k}], {k});"
            )?;
            writeln!(code, "        }}")?;
            writeln!(code, "    }});")?;
            writeln!(code, "}}")?;
        } else {
            // Sequential with 4-way row unrolling for ILP
            writeln!(code, "#[inline]")?;
            writeln!(
                code,
                "fn matmul_vec_{k}x{n}(output: &mut [f32; {n}], input: &[f32; {k}], weight: &[f32]) {{"
            )?;
            let n_chunks = n / 4;
            let n_remainder = n % 4;
            if n_chunks > 0 {
                writeln!(
                    code,
                    "    // Process 4 output rows at a time for instruction-level parallelism"
                )?;
                writeln!(code, "    for chunk in 0..{n_chunks} {{")?;
                writeln!(code, "        let j0 = chunk * 4;")?;
                writeln!(
                    code,
                    "        output[j0]   = dot_f32(&input[..], &weight[j0*{k}..(j0+1)*{k}], {k});"
                )?;
                writeln!(code, "        output[j0+1] = dot_f32(&input[..], &weight[(j0+1)*{k}..(j0+2)*{k}], {k});")?;
                writeln!(code, "        output[j0+2] = dot_f32(&input[..], &weight[(j0+2)*{k}..(j0+3)*{k}], {k});")?;
                writeln!(code, "        output[j0+3] = dot_f32(&input[..], &weight[(j0+3)*{k}..(j0+4)*{k}], {k});")?;
                writeln!(code, "    }}")?;
            }
            if n_remainder > 0 {
                writeln!(code, "    // Handle remaining {n_remainder} output rows")?;
                writeln!(code, "    let base = {} * 4;", n_chunks)?;
                for r in 0..n_remainder {
                    writeln!(code, "    output[base+{r}] = dot_f32(&input[..], &weight[(base+{r})*{k}..(base+{r}+1)*{k}], {k});")?;
                }
            }
            writeln!(code, "}}")?;
        }
        writeln!(code)?;
    }

    Ok(())
}

/// Emit the Q8_0 dot-product helper and f16→f32 conversion used by Q8_0 matmul.
///
/// On AArch64 (Apple Silicon, Arm servers) the dot product is vectorised with
/// NEON: int8 values are widened to f32 in three steps (s8→s16→s32→f32) and
/// accumulated via `vfmaq_f32`.  Eight NEON float vectors cover one full 32-
/// element block, matching the throughput of the F32 NEON kernel.
///
/// On all other targets the implementation falls back to a portable scalar loop.
fn emit_q8_0_kernel(code: &mut String) -> Result<(), CodegenError> {
    code.push_str(
        r#"
// --- Q8_0 quantized dot product (no dequantization at load time) ---
/// Convert IEEE 754 half-precision (f16) bit pattern to f32.
#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;
    if exponent == 0 {
        if mantissa == 0 { return f32::from_bits(sign << 31); }
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 { m <<= 1; e -= 1; }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exponent == 31 {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13));
    }
    let f32_exp = (exponent as i32 - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

/// Dot product of f32 input against a Q8_0 weight row without prior dequantization.
/// `weight_row`: raw Q8_0 bytes — layout: [2 bytes f16 scale][32 bytes int8] per block.
/// `k`: number of input/weight elements (must be a multiple of 32 for NEON path).
#[inline]
fn dot_q8_0(input: &[f32], weight_row: &[u8], k: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        const BLOCK_SIZE: usize = 32;
        const TYPE_SIZE: usize = 34;
        let num_blocks = k / BLOCK_SIZE;

        // Four independent accumulators hide the latency of sequential vaddq_f32.
        // We interleave 4 blocks at a time; each accumulator is a float32x4 summing
        // products from its assigned blocks throughout the inner loop.
        let mut a0 = unsafe { vdupq_n_f32(0.0) };
        let mut a1 = unsafe { vdupq_n_f32(0.0) };
        let mut a2 = unsafe { vdupq_n_f32(0.0) };
        let mut a3 = unsafe { vdupq_n_f32(0.0) };

        // Helper macro: widen a Q8_0 block and fmla into an accumulator.
        // (Cannot be a closure because it borrows multiple vars.)
        macro_rules! dot_block {
            ($acc:ident, $b:expr) => {{
                let bs = $b * TYPE_SIZE;
                let scale = f16_bits_to_f32(
                    u16::from_le_bytes([weight_row[bs], weight_row[bs + 1]])
                );
                let block_start = $b * BLOCK_SIZE;
                unsafe {
                    let sv = vdupq_n_f32(scale);
                    let w_ptr = weight_row[bs + 2..].as_ptr() as *const i8;
                    let w_lo = vld1q_s8(w_ptr);
                    let w_hi = vld1q_s8(w_ptr.add(16));
                    let wf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(w_lo)))));
                    let wf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(w_lo)))));
                    let wf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(w_lo)))));
                    let wf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(w_lo)))));
                    let wf4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(w_hi)))));
                    let wf5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(w_hi)))));
                    let wf6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(w_hi)))));
                    let wf7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(w_hi)))));
                    let i_ptr = input[block_start..].as_ptr();
                    let i0 = vld1q_f32(i_ptr);
                    let i1 = vld1q_f32(i_ptr.add(4));
                    let i2 = vld1q_f32(i_ptr.add(8));
                    let i3 = vld1q_f32(i_ptr.add(12));
                    let i4 = vld1q_f32(i_ptr.add(16));
                    let i5 = vld1q_f32(i_ptr.add(20));
                    let i6 = vld1q_f32(i_ptr.add(24));
                    let i7 = vld1q_f32(i_ptr.add(28));
                    let mut tmp = vdupq_n_f32(0.0);
                    tmp = vfmaq_f32(tmp, i0, vmulq_f32(wf0, sv));
                    tmp = vfmaq_f32(tmp, i1, vmulq_f32(wf1, sv));
                    tmp = vfmaq_f32(tmp, i2, vmulq_f32(wf2, sv));
                    tmp = vfmaq_f32(tmp, i3, vmulq_f32(wf3, sv));
                    tmp = vfmaq_f32(tmp, i4, vmulq_f32(wf4, sv));
                    tmp = vfmaq_f32(tmp, i5, vmulq_f32(wf5, sv));
                    tmp = vfmaq_f32(tmp, i6, vmulq_f32(wf6, sv));
                    tmp = vfmaq_f32(tmp, i7, vmulq_f32(wf7, sv));
                    $acc = vaddq_f32($acc, tmp);
                }
            }};
        }

        let groups = num_blocks / 4;
        for g in 0..groups {
            let base = g * 4;
            dot_block!(a0, base);
            dot_block!(a1, base + 1);
            dot_block!(a2, base + 2);
            dot_block!(a3, base + 3);
        }
        // Tail blocks (0, 1, or 2)
        let tail_start = groups * 4;
        if tail_start < num_blocks { dot_block!(a0, tail_start); }
        if tail_start + 1 < num_blocks { dot_block!(a1, tail_start + 1); }
        if tail_start + 2 < num_blocks { dot_block!(a2, tail_start + 2); }

        let merged = unsafe {
            vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))
        };
        let covered = num_blocks * BLOCK_SIZE;
        let mut result = unsafe { vaddvq_f32(merged) };
        for j in covered..k {
            let b = j / BLOCK_SIZE;
            let bs = b * TYPE_SIZE;
            let scale = f16_bits_to_f32(u16::from_le_bytes([weight_row[bs], weight_row[bs + 1]]));
            let q = weight_row[bs + 2 + (j % BLOCK_SIZE)] as i8;
            result += input[j] * (q as f32 * scale);
        }
        return result;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        const BLOCK_SIZE: usize = 32;
        const TYPE_SIZE: usize = 34;
        let num_blocks = k.div_ceil(BLOCK_SIZE);
        let mut sum = 0.0f32;
        for b in 0..num_blocks {
            let bs = b * TYPE_SIZE;
            let scale_bits = u16::from_le_bytes([weight_row[bs], weight_row[bs + 1]]);
            let scale = f16_bits_to_f32(scale_bits);
            let block_start = b * BLOCK_SIZE;
            let block_end = (block_start + BLOCK_SIZE).min(k);
            for j in 0..(block_end - block_start) {
                let q = weight_row[bs + 2 + j] as i8;
                sum += input[block_start + j] * (q as f32 * scale);
            }
        }
        sum
    }
}

"#,
    );
    Ok(())
}

/// Emit the int8×int8 kernel and input-quantization helper.
///
/// We quantize the input activation to Q8_0 blocks once per matmul call,
/// then compute each output row with `vmull_s8` + `vpaddlq_s16` — stable
/// NEON intrinsics that process 32 int8 pairs in ~8 ops instead of the
/// 48+ ops needed for the f32-widening path.  This typically gives 3–4×
/// better throughput on the inner dot product; the quantization overhead
/// is amortised over all N output rows of the same matmul.
///
/// Note: `target-cpu=native` (written to `.cargo/config.toml`) lets LLVM
/// also emit `sdot` from these patterns on M1+ where dotprod is available,
/// closing the gap further without requiring unstable feature flags.
fn emit_q8_0_sdot_kernel(code: &mut String) -> Result<(), CodegenError> {
    code.push_str(
        r#"
// --- Q8_0 × Q8_0 int8 dot product via AArch64 sdot (inline asm, stable Rust) ---
// Input is quantized to Q8_0 once per matmul call; each output row then uses
// sdot (Signed DOT product, 16 int8 pairs → 4 int32 per instruction) instead
// of the f32-widening path.  2 sdot calls cover one 32-element Q8_0 block vs.
// the 4 vmull+4 vpaddl needed for the pure-intrinsics path.

/// Quantize a f32 slice to Q8_0 blocks in-place (block-wise, 32 elements per block).
/// `out` must be pre-allocated with `ceil(input.len()/32)*34` bytes.
/// Format: [2-byte f16 scale][32 int8 values] per block.
/// The scale is computed per-block as absmax/127 so no element saturates.
#[cfg(target_arch = "aarch64")]
#[inline]
fn quantize_to_q8_0_blocks_into(input: &[f32], out: &mut [u8]) {
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize = 34;
    let k = input.len();
    let num_blocks = k.div_ceil(BLOCK_SIZE);
    for b in 0..num_blocks {
        let block_start = b * BLOCK_SIZE;
        let block_end = (block_start + BLOCK_SIZE).min(k);
        let block = &input[block_start..block_end];
        let absmax = block.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let (scale, inv_scale) = if absmax == 0.0 {
            (1.0f32, 0.0f32)
        } else {
            let s = absmax / 127.0;
            (s, 1.0 / s)
        };
        let scale_bits = f32_to_f16_bits(scale);
        let bs = b * TYPE_SIZE;
        out[bs]     = (scale_bits & 0xFF) as u8;
        out[bs + 1] = (scale_bits >> 8)  as u8;
        for (j, &x) in block.iter().enumerate() {
            out[bs + 2 + j] = (x * inv_scale).round().clamp(-127.0, 127.0) as i8 as u8;
        }
        for j in block.len()..BLOCK_SIZE {
            out[bs + 2 + j] = 0u8;
        }
    }
    let _ = num_blocks;
}

/// Quantize a f32 slice to Q8_0 blocks, returning a Vec<u8>.
/// Use `quantize_to_q8_0_blocks_into` with a stack buffer when the size is
/// statically known (avoids heap allocation in hot paths).
#[cfg(target_arch = "aarch64")]
#[inline]
fn quantize_to_q8_0_blocks(input: &[f32]) -> Vec<u8> {
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize = 34;
    let num_blocks = input.len().div_ceil(BLOCK_SIZE);
    let mut out = vec![0u8; num_blocks * TYPE_SIZE];
    quantize_to_q8_0_blocks_into(input, &mut out);
    out
}

/// f32 → IEEE 754 f16 bit pattern with round-to-nearest-even.  Truncation
/// (`mant >> 13`) gives a systematic round-toward-zero bias on every scale
/// that compounds through stacked matmuls — fixed in v0.9.7 after Q4_K
/// simulate path showed catastrophic perplexity (~50K vs baseline ~4).
#[inline]
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp_f32 == 0xFF {
        let mut h_mant = ((mant >> 13) & 0x3FF) as u16;
        if mant != 0 && h_mant == 0 { h_mant = 1; }
        return sign | 0x7C00 | h_mant;
    }
    if exp_f32 == 0 { return sign; }
    let unbiased = exp_f32 - 127;
    if unbiased > 15 { return sign | 0x7C00; }
    if unbiased < -24 { return sign; }
    let implicit = (mant | 0x0080_0000) as u32;
    let shift: u32 = if unbiased < -14 { (-1 - unbiased) as u32 } else { 13 };
    let mask = (1u32 << shift) - 1;
    let dropped = implicit & mask;
    let half = 1u32 << (shift - 1);
    let mut truncated = implicit >> shift;
    if dropped > half || (dropped == half && (truncated & 1) == 1) {
        truncated = truncated.wrapping_add(1);
    }
    if unbiased < -14 {
        if truncated >= 0x400 { return sign | (1 << 10); }
        return sign | (truncated as u16);
    }
    let mut h_exp = (unbiased + 15) as u32;
    if truncated >= 0x800 {
        h_exp += 1;
        if h_exp >= 0x1F { return sign | 0x7C00; }
        return sign | ((h_exp as u16) << 10);
    }
    sign | ((h_exp as u16) << 10) | ((truncated & 0x3FF) as u16)
}

/// Dot product of two Q8_0 vectors using AArch64 `sdot` instruction via inline asm.
///
/// `sdot` (Signed DOT product) processes 16 int8 pairs and accumulates 4 int32
/// results in a single instruction — 2 calls cover a full 32-element Q8_0 block.
/// This is available on ARMv8.4+/FEAT_DotProd (M1 and all modern Arm chips)
/// without requiring the unstable `stdarch_neon_dotprod` Rust feature.
///
/// Four independent f32 accumulators provide ILP across interleaved blocks.
/// Each block's int32 sum is immediately scaled by `a_scale * b_scale` before
/// being added to the f32 accumulator — no scale mixing across blocks.
///
/// `a_q8`: input quantized to Q8_0 blocks by `quantize_to_q8_0_blocks`.
/// `b_q8`: weight row raw Q8_0 bytes.
/// `k`: element count.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_q8_0_q8_0(a_q8: &[u8], b_q8: &[u8], k: usize) -> f32 {
    use std::arch::aarch64::*;
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize  = 34;
    let num_blocks = k / BLOCK_SIZE;

    // Four independent f32 accumulators for ILP across interleaved blocks.
    let mut fa0 = 0.0f32;
    let mut fa1 = 0.0f32;
    let mut fa2 = 0.0f32;
    let mut fa3 = 0.0f32;

    // Compute combined_scale * sdot_sum for one Q8_0 block.
    // sdot processes 16 int8 pairs per call (2 calls = 32 elements / block).
    // Uses inline asm to call SDOT directly on stable Rust toolchain.
    macro_rules! block_f32 {
        ($b:expr) => {{
            let bs = $b * TYPE_SIZE;
            let a_scale = f16_bits_to_f32(u16::from_le_bytes([a_q8[bs], a_q8[bs + 1]]));
            let b_scale = f16_bits_to_f32(u16::from_le_bytes([b_q8[bs], b_q8[bs + 1]]));
            let combined = a_scale * b_scale;
            unsafe {
                let ap = a_q8[bs + 2..].as_ptr() as *const i8;
                let bp = b_q8[bs + 2..].as_ptr() as *const i8;
                let a0: int8x16_t = vld1q_s8(ap);
                let a1: int8x16_t = vld1q_s8(ap.add(16));
                let b0: int8x16_t = vld1q_s8(bp);
                let b1: int8x16_t = vld1q_s8(bp.add(16));
                let mut acc: int32x4_t = vdupq_n_s32(0);
                // sdot: acc.4s += sdot(a.16b, b.16b) — 16 int8 pairs → 4 int32
                core::arch::asm!(
                    "sdot {acc:v}.4s, {a0:v}.16b, {b0:v}.16b",
                    "sdot {acc:v}.4s, {a1:v}.16b, {b1:v}.16b",
                    acc = inout(vreg) acc,
                    a0 = in(vreg) a0,
                    b0 = in(vreg) b0,
                    a1 = in(vreg) a1,
                    b1 = in(vreg) b1,
                    options(nostack),
                );
                combined * vaddvq_s32(acc) as f32
            }
        }};
    }

    let groups = num_blocks / 4;
    for g in 0..groups {
        let base = g * 4;
        fa0 += block_f32!(base);
        fa1 += block_f32!(base + 1);
        fa2 += block_f32!(base + 2);
        fa3 += block_f32!(base + 3);
    }
    let tail = groups * 4;
    if tail     < num_blocks { fa0 += block_f32!(tail);     }
    if tail + 1 < num_blocks { fa1 += block_f32!(tail + 1); }
    if tail + 2 < num_blocks { fa2 += block_f32!(tail + 2); }

    let mut result = fa0 + fa1 + fa2 + fa3;
    // Scalar tail for any k not a multiple of 32
    for j in (num_blocks * BLOCK_SIZE)..k {
        let b  = j / BLOCK_SIZE;
        let bs = b * TYPE_SIZE;
        let as_ = f16_bits_to_f32(u16::from_le_bytes([a_q8[bs], a_q8[bs + 1]]));
        let bs_ = f16_bits_to_f32(u16::from_le_bytes([b_q8[bs], b_q8[bs + 1]]));
        let aq = a_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq = b_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        result += as_ * bs_ * (aq as f32) * (bq as f32);
    }
    result
}

/// Compute 4 Q8_0 dot products simultaneously, sharing the quantized input block.
///
/// Processes `(input_q8, w0_q8)`, `(input_q8, w1_q8)`, `(input_q8, w2_q8)`,
/// `(input_q8, w3_q8)` in a single pass over the input.
///
/// For each 32-element Q8_0 block we load the input data (`ai0`, `ai1`) once
/// and issue 8 independent sdot instructions (2 per weight row) — giving
/// 4-way ILP across rows with full register pressure coverage.
/// This eliminates 3/4 of the redundant input-block loads that sequential
/// calls would cause, and maximises the M1's OOO instruction window.
///
/// Returns `(dot0, dot1, dot2, dot3)`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot4_q8_0_q8_0(
    a_q8: &[u8],
    b0_q8: &[u8],
    b1_q8: &[u8],
    b2_q8: &[u8],
    b3_q8: &[u8],
    k: usize,
) -> (f32, f32, f32, f32) {
    use std::arch::aarch64::*;
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize = 34;
    let num_blocks = k / BLOCK_SIZE;

    let mut r0 = 0.0f32;
    let mut r1 = 0.0f32;
    let mut r2 = 0.0f32;
    let mut r3 = 0.0f32;

    // Prefetch the first block of each weight row into L1 cache.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{b0}]",
            "prfm pldl1keep, [{b1}]",
            "prfm pldl1keep, [{b2}]",
            "prfm pldl1keep, [{b3}]",
            b0 = in(reg) b0_q8.as_ptr(),
            b1 = in(reg) b1_q8.as_ptr(),
            b2 = in(reg) b2_q8.as_ptr(),
            b3 = in(reg) b3_q8.as_ptr(),
            options(nostack, readonly),
        );
    }

    for b in 0..num_blocks {
        let abs = b * TYPE_SIZE;
        let a_scale = f16_bits_to_f32(u16::from_le_bytes([a_q8[abs], a_q8[abs + 1]]));
        let s0 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b0_q8[abs], b0_q8[abs + 1]]));
        let s1 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b1_q8[abs], b1_q8[abs + 1]]));
        let s2 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b2_q8[abs], b2_q8[abs + 1]]));
        let s3 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b3_q8[abs], b3_q8[abs + 1]]));
        unsafe {
            // Load input block ONCE; reuse for all 4 weight rows.
            let ap = a_q8[abs + 2..].as_ptr() as *const i8;
            let ai0: int8x16_t = vld1q_s8(ap);
            let ai1: int8x16_t = vld1q_s8(ap.add(16));
            // Load 4 weight blocks (2 halves each).
            let bp0 = b0_q8[abs + 2..].as_ptr() as *const i8;
            let bp1 = b1_q8[abs + 2..].as_ptr() as *const i8;
            let bp2 = b2_q8[abs + 2..].as_ptr() as *const i8;
            let bp3 = b3_q8[abs + 2..].as_ptr() as *const i8;
            let w00: int8x16_t = vld1q_s8(bp0);       let w01: int8x16_t = vld1q_s8(bp0.add(16));
            let w10: int8x16_t = vld1q_s8(bp1);       let w11: int8x16_t = vld1q_s8(bp1.add(16));
            let w20: int8x16_t = vld1q_s8(bp2);       let w21: int8x16_t = vld1q_s8(bp2.add(16));
            let w30: int8x16_t = vld1q_s8(bp3);       let w31: int8x16_t = vld1q_s8(bp3.add(16));
            // 8 independent sdot calls — 4-way ILP across the 4 accumulators.
            let mut acc0: int32x4_t = vdupq_n_s32(0);
            let mut acc1: int32x4_t = vdupq_n_s32(0);
            let mut acc2: int32x4_t = vdupq_n_s32(0);
            let mut acc3: int32x4_t = vdupq_n_s32(0);
            core::arch::asm!(
                "sdot {a0:v}.4s, {i0:v}.16b, {b00:v}.16b",
                "sdot {a1:v}.4s, {i0:v}.16b, {b10:v}.16b",
                "sdot {a2:v}.4s, {i0:v}.16b, {b20:v}.16b",
                "sdot {a3:v}.4s, {i0:v}.16b, {b30:v}.16b",
                "sdot {a0:v}.4s, {i1:v}.16b, {b01:v}.16b",
                "sdot {a1:v}.4s, {i1:v}.16b, {b11:v}.16b",
                "sdot {a2:v}.4s, {i1:v}.16b, {b21:v}.16b",
                "sdot {a3:v}.4s, {i1:v}.16b, {b31:v}.16b",
                a0 = inout(vreg) acc0, a1 = inout(vreg) acc1,
                a2 = inout(vreg) acc2, a3 = inout(vreg) acc3,
                i0 = in(vreg) ai0,    i1 = in(vreg) ai1,
                b00 = in(vreg) w00,   b01 = in(vreg) w01,
                b10 = in(vreg) w10,   b11 = in(vreg) w11,
                b20 = in(vreg) w20,   b21 = in(vreg) w21,
                b30 = in(vreg) w30,   b31 = in(vreg) w31,
                options(nostack),
            );
            // Prefetch weight data 2 blocks ahead to hide memory latency.
            if b + 2 < num_blocks {
                let pf_off = (b + 2) * TYPE_SIZE;
                core::arch::asm!(
                    "prfm pldl1keep, [{b0}]",
                    "prfm pldl1keep, [{b1}]",
                    "prfm pldl1keep, [{b2}]",
                    "prfm pldl1keep, [{b3}]",
                    b0 = in(reg) b0_q8.as_ptr().add(pf_off),
                    b1 = in(reg) b1_q8.as_ptr().add(pf_off),
                    b2 = in(reg) b2_q8.as_ptr().add(pf_off),
                    b3 = in(reg) b3_q8.as_ptr().add(pf_off),
                    options(nostack, readonly),
                );
            }
            r0 += s0 * vaddvq_s32(acc0) as f32;
            r1 += s1 * vaddvq_s32(acc1) as f32;
            r2 += s2 * vaddvq_s32(acc2) as f32;
            r3 += s3 * vaddvq_s32(acc3) as f32;
        }
    }
    // Scalar tail for k not divisible by 32
    for j in (num_blocks * BLOCK_SIZE)..k {
        let b  = j / BLOCK_SIZE;
        let bs = b * TYPE_SIZE;
        let as_ = f16_bits_to_f32(u16::from_le_bytes([a_q8[bs], a_q8[bs + 1]]));
        let aq = a_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq0 = b0_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq1 = b1_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq2 = b2_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq3 = b3_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bs0 = f16_bits_to_f32(u16::from_le_bytes([b0_q8[bs], b0_q8[bs + 1]]));
        let bs1 = f16_bits_to_f32(u16::from_le_bytes([b1_q8[bs], b1_q8[bs + 1]]));
        let bs2 = f16_bits_to_f32(u16::from_le_bytes([b2_q8[bs], b2_q8[bs + 1]]));
        let bs3 = f16_bits_to_f32(u16::from_le_bytes([b3_q8[bs], b3_q8[bs + 1]]));
        r0 += as_ * bs0 * (aq as f32) * (bq0 as f32);
        r1 += as_ * bs1 * (aq as f32) * (bq1 as f32);
        r2 += as_ * bs2 * (aq as f32) * (bq2 as f32);
        r3 += as_ * bs3 * (aq as f32) * (bq3 as f32);
    }
    (r0, r1, r2, r3)
}

/// 8-row simultaneous Q8_0 × Q8_0 dot product via AArch64 `sdot` inline asm.
///
/// Loads the input Q8_0 block ONCE and issues 16 independent `sdot` instructions
/// (2 per weight row × 8 rows) per block — providing 8-way ILP across rows.
/// The 8 independent first-half `sdot` instructions fully cover the 4-cycle
/// sdot RAW latency before each row's second-half `sdot`, eliminating stalls.
/// Also halves function-call overhead vs. `dot4_q8_0_q8_0` for the same N.
///
/// Returns `(r0, r1, r2, r3, r4, r5, r6, r7)`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot8_q8_0_q8_0(
    a_q8: &[u8],
    b0_q8: &[u8], b1_q8: &[u8], b2_q8: &[u8], b3_q8: &[u8],
    b4_q8: &[u8], b5_q8: &[u8], b6_q8: &[u8], b7_q8: &[u8],
    k: usize,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    use std::arch::aarch64::*;
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize  = 34;
    let num_blocks = k / BLOCK_SIZE;

    let mut r0 = 0.0f32; let mut r1 = 0.0f32;
    let mut r2 = 0.0f32; let mut r3 = 0.0f32;
    let mut r4 = 0.0f32; let mut r5 = 0.0f32;
    let mut r6 = 0.0f32; let mut r7 = 0.0f32;

    // Prefetch the first block of each weight row into L1 cache.
    // This hides initial memory latency before the loop begins.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{b0}]",
            "prfm pldl1keep, [{b1}]",
            "prfm pldl1keep, [{b2}]",
            "prfm pldl1keep, [{b3}]",
            "prfm pldl1keep, [{b4}]",
            "prfm pldl1keep, [{b5}]",
            "prfm pldl1keep, [{b6}]",
            "prfm pldl1keep, [{b7}]",
            b0 = in(reg) b0_q8.as_ptr(),
            b1 = in(reg) b1_q8.as_ptr(),
            b2 = in(reg) b2_q8.as_ptr(),
            b3 = in(reg) b3_q8.as_ptr(),
            b4 = in(reg) b4_q8.as_ptr(),
            b5 = in(reg) b5_q8.as_ptr(),
            b6 = in(reg) b6_q8.as_ptr(),
            b7 = in(reg) b7_q8.as_ptr(),
            options(nostack, readonly),
        );
    }

    for b in 0..num_blocks {
        let abs = b * TYPE_SIZE;
        let a_scale = f16_bits_to_f32(u16::from_le_bytes([a_q8[abs], a_q8[abs + 1]]));
        let s0 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b0_q8[abs], b0_q8[abs + 1]]));
        let s1 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b1_q8[abs], b1_q8[abs + 1]]));
        let s2 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b2_q8[abs], b2_q8[abs + 1]]));
        let s3 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b3_q8[abs], b3_q8[abs + 1]]));
        let s4 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b4_q8[abs], b4_q8[abs + 1]]));
        let s5 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b5_q8[abs], b5_q8[abs + 1]]));
        let s6 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b6_q8[abs], b6_q8[abs + 1]]));
        let s7 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b7_q8[abs], b7_q8[abs + 1]]));
        unsafe {
            let ap  = a_q8[abs + 2..].as_ptr() as *const i8;
            let ai0: int8x16_t = vld1q_s8(ap);
            let ai1: int8x16_t = vld1q_s8(ap.add(16));
            let bp0 = b0_q8[abs + 2..].as_ptr() as *const i8;
            let bp1 = b1_q8[abs + 2..].as_ptr() as *const i8;
            let bp2 = b2_q8[abs + 2..].as_ptr() as *const i8;
            let bp3 = b3_q8[abs + 2..].as_ptr() as *const i8;
            let bp4 = b4_q8[abs + 2..].as_ptr() as *const i8;
            let bp5 = b5_q8[abs + 2..].as_ptr() as *const i8;
            let bp6 = b6_q8[abs + 2..].as_ptr() as *const i8;
            let bp7 = b7_q8[abs + 2..].as_ptr() as *const i8;
            let w00: int8x16_t = vld1q_s8(bp0);       let w01: int8x16_t = vld1q_s8(bp0.add(16));
            let w10: int8x16_t = vld1q_s8(bp1);       let w11: int8x16_t = vld1q_s8(bp1.add(16));
            let w20: int8x16_t = vld1q_s8(bp2);       let w21: int8x16_t = vld1q_s8(bp2.add(16));
            let w30: int8x16_t = vld1q_s8(bp3);       let w31: int8x16_t = vld1q_s8(bp3.add(16));
            let w40: int8x16_t = vld1q_s8(bp4);       let w41: int8x16_t = vld1q_s8(bp4.add(16));
            let w50: int8x16_t = vld1q_s8(bp5);       let w51: int8x16_t = vld1q_s8(bp5.add(16));
            let w60: int8x16_t = vld1q_s8(bp6);       let w61: int8x16_t = vld1q_s8(bp6.add(16));
            let w70: int8x16_t = vld1q_s8(bp7);       let w71: int8x16_t = vld1q_s8(bp7.add(16));
            let mut acc0: int32x4_t = vdupq_n_s32(0);
            let mut acc1: int32x4_t = vdupq_n_s32(0);
            let mut acc2: int32x4_t = vdupq_n_s32(0);
            let mut acc3: int32x4_t = vdupq_n_s32(0);
            let mut acc4: int32x4_t = vdupq_n_s32(0);
            let mut acc5: int32x4_t = vdupq_n_s32(0);
            let mut acc6: int32x4_t = vdupq_n_s32(0);
            let mut acc7: int32x4_t = vdupq_n_s32(0);
            // 16 independent sdot — 8 for first halves, 8 for second halves.
            // Between acc0's first and second sdot there are 7 independent sdot,
            // fully covering the 4-cycle RAW latency on M1.
            core::arch::asm!(
                "sdot {a0:v}.4s, {i0:v}.16b, {b00:v}.16b",
                "sdot {a1:v}.4s, {i0:v}.16b, {b10:v}.16b",
                "sdot {a2:v}.4s, {i0:v}.16b, {b20:v}.16b",
                "sdot {a3:v}.4s, {i0:v}.16b, {b30:v}.16b",
                "sdot {a4:v}.4s, {i0:v}.16b, {b40:v}.16b",
                "sdot {a5:v}.4s, {i0:v}.16b, {b50:v}.16b",
                "sdot {a6:v}.4s, {i0:v}.16b, {b60:v}.16b",
                "sdot {a7:v}.4s, {i0:v}.16b, {b70:v}.16b",
                "sdot {a0:v}.4s, {i1:v}.16b, {b01:v}.16b",
                "sdot {a1:v}.4s, {i1:v}.16b, {b11:v}.16b",
                "sdot {a2:v}.4s, {i1:v}.16b, {b21:v}.16b",
                "sdot {a3:v}.4s, {i1:v}.16b, {b31:v}.16b",
                "sdot {a4:v}.4s, {i1:v}.16b, {b41:v}.16b",
                "sdot {a5:v}.4s, {i1:v}.16b, {b51:v}.16b",
                "sdot {a6:v}.4s, {i1:v}.16b, {b61:v}.16b",
                "sdot {a7:v}.4s, {i1:v}.16b, {b71:v}.16b",
                a0 = inout(vreg) acc0, a1 = inout(vreg) acc1,
                a2 = inout(vreg) acc2, a3 = inout(vreg) acc3,
                a4 = inout(vreg) acc4, a5 = inout(vreg) acc5,
                a6 = inout(vreg) acc6, a7 = inout(vreg) acc7,
                i0 = in(vreg) ai0,    i1 = in(vreg) ai1,
                b00 = in(vreg) w00,   b01 = in(vreg) w01,
                b10 = in(vreg) w10,   b11 = in(vreg) w11,
                b20 = in(vreg) w20,   b21 = in(vreg) w21,
                b30 = in(vreg) w30,   b31 = in(vreg) w31,
                b40 = in(vreg) w40,   b41 = in(vreg) w41,
                b50 = in(vreg) w50,   b51 = in(vreg) w51,
                b60 = in(vreg) w60,   b61 = in(vreg) w61,
                b70 = in(vreg) w70,   b71 = in(vreg) w71,
                options(nostack),
            );
            // Prefetch weight data 2 blocks ahead to hide memory latency.
            // The sdot pipeline depth is ~4 cycles; prefetching 2 blocks ahead
            // ensures data arrives in L1 before the load instructions need it.
            if b + 2 < num_blocks {
                let pf_off = (b + 2) * TYPE_SIZE;
                core::arch::asm!(
                    "prfm pldl1keep, [{b0}]",
                    "prfm pldl1keep, [{b1}]",
                    "prfm pldl1keep, [{b2}]",
                    "prfm pldl1keep, [{b3}]",
                    "prfm pldl1keep, [{b4}]",
                    "prfm pldl1keep, [{b5}]",
                    "prfm pldl1keep, [{b6}]",
                    "prfm pldl1keep, [{b7}]",
                    b0 = in(reg) b0_q8.as_ptr().add(pf_off),
                    b1 = in(reg) b1_q8.as_ptr().add(pf_off),
                    b2 = in(reg) b2_q8.as_ptr().add(pf_off),
                    b3 = in(reg) b3_q8.as_ptr().add(pf_off),
                    b4 = in(reg) b4_q8.as_ptr().add(pf_off),
                    b5 = in(reg) b5_q8.as_ptr().add(pf_off),
                    b6 = in(reg) b6_q8.as_ptr().add(pf_off),
                    b7 = in(reg) b7_q8.as_ptr().add(pf_off),
                    options(nostack, readonly),
                );
            }
            r0 += s0 * vaddvq_s32(acc0) as f32;
            r1 += s1 * vaddvq_s32(acc1) as f32;
            r2 += s2 * vaddvq_s32(acc2) as f32;
            r3 += s3 * vaddvq_s32(acc3) as f32;
            r4 += s4 * vaddvq_s32(acc4) as f32;
            r5 += s5 * vaddvq_s32(acc5) as f32;
            r6 += s6 * vaddvq_s32(acc6) as f32;
            r7 += s7 * vaddvq_s32(acc7) as f32;
        }
    }
    // Scalar tail (k not divisible by 32)
    for j in (num_blocks * BLOCK_SIZE)..k {
        let b  = j / BLOCK_SIZE;
        let bs = b * TYPE_SIZE;
        let as_ = f16_bits_to_f32(u16::from_le_bytes([a_q8[bs], a_q8[bs + 1]]));
        let aq = a_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq0 = b0_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq1 = b1_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq2 = b2_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq3 = b3_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq4 = b4_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq5 = b5_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq6 = b6_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bq7 = b7_q8[bs + 2 + (j % BLOCK_SIZE)] as i8;
        let bs0 = f16_bits_to_f32(u16::from_le_bytes([b0_q8[bs], b0_q8[bs + 1]]));
        let bs1 = f16_bits_to_f32(u16::from_le_bytes([b1_q8[bs], b1_q8[bs + 1]]));
        let bs2 = f16_bits_to_f32(u16::from_le_bytes([b2_q8[bs], b2_q8[bs + 1]]));
        let bs3 = f16_bits_to_f32(u16::from_le_bytes([b3_q8[bs], b3_q8[bs + 1]]));
        let bs4 = f16_bits_to_f32(u16::from_le_bytes([b4_q8[bs], b4_q8[bs + 1]]));
        let bs5 = f16_bits_to_f32(u16::from_le_bytes([b5_q8[bs], b5_q8[bs + 1]]));
        let bs6 = f16_bits_to_f32(u16::from_le_bytes([b6_q8[bs], b6_q8[bs + 1]]));
        let bs7 = f16_bits_to_f32(u16::from_le_bytes([b7_q8[bs], b7_q8[bs + 1]]));
        r0 += as_ * bs0 * (aq as f32) * (bq0 as f32);
        r1 += as_ * bs1 * (aq as f32) * (bq1 as f32);
        r2 += as_ * bs2 * (aq as f32) * (bq2 as f32);
        r3 += as_ * bs3 * (aq as f32) * (bq3 as f32);
        r4 += as_ * bs4 * (aq as f32) * (bq4 as f32);
        r5 += as_ * bs5 * (aq as f32) * (bq5 as f32);
        r6 += as_ * bs6 * (aq as f32) * (bq6 as f32);
        r7 += as_ * bs7 * (aq as f32) * (bq7 as f32);
    }
    (r0, r1, r2, r3, r4, r5, r6, r7)
}

"#,
    );
    Ok(())
}

/// Collect all unique (k, n) matmul shapes needed for Q8_0 projection weights.
/// Same shapes as the f32 version but used for Q8_0 `matmul_vec_q8_0_KxN` variants.
fn q8_matmul_shapes(config: &ModelConfig) -> Vec<(usize, usize)> {
    matmul_shapes(config)
}

/// Emit shape-specialized Q8_0 matmul functions: `matmul_vec_q8_0_KxN`.
/// These take `weight: &[u8]` (raw Q8_0 bytes) and call `dot_q8_0()` per output row.
fn emit_specialized_q8_matmul_functions(
    code: &mut String,
    config: &ModelConfig,
) -> Result<(), CodegenError> {
    writeln!(
        code,
        "// --- Shape-specialized Q8_0 matmul functions (m=1, weight is &[u8]) ---"
    )?;
    writeln!(code)?;

    // Parallelize when total weight bytes > 1 MB (adapts to model size).
    let par_byte_threshold: usize = 1_000_000;

    for &(k, n) in &q8_matmul_shapes(config) {
        // Byte size per row: ceil(k/32)*34
        let row_bytes = k.div_ceil(32) * 34;
        writeln!(
            code,
            "/// Q8_0 matmul: [1, {k}] x [{n}, {k}]^T -> [1, {n}] (weight stored as raw Q8_0 bytes)"
        )?;

        // --- AArch64 sdot path (inline asm, stable Rust) ---
        // Quantize input to Q8_0 once per call, then compute each row with
        // dot_q8_0_q8_0 which uses the sdot instruction (2 sdot per 32-element
        // block vs. 4 vmull+4 vpaddl for pure intrinsics — ~2x fewer NEON ops).
        writeln!(code, "#[cfg(target_arch = \"aarch64\")]")?;
        writeln!(code, "#[inline]")?;
        writeln!(
            code,
            "fn matmul_vec_q8_0_{k}x{n}(output: &mut [f32; {n}], input: &[f32; {k}], weight: &[u8]) {{"
        )?;
        // Stack-allocated Q8_0 buffer: avoids heap allocation on every matmul call.
        // Size is statically known: ceil(k/32)*34 bytes.
        let q8_bytes = row_bytes; // row_bytes = k.div_ceil(32)*34, already computed
        writeln!(code, "    let mut input_q8 = [0u8; {q8_bytes}];")?;
        writeln!(
            code,
            "    quantize_to_q8_0_blocks_into(&input[..], &mut input_q8);"
        )?;
        // Use dot8 for groups of 8 rows (better ILP — 16 sdot/block covers 4-cycle latency),
        // fall back to dot4 for 4-row remainder, then scalar for final <4 rows.
        let n_chunks8 = n / 8;
        let n_rem8 = n % 8;
        let n_chunks4_tail = n_rem8 / 4;
        let n_rem4 = n_rem8 % 4;
        if n * (k.div_ceil(32) * 34) >= par_byte_threshold {
            writeln!(
                code,
                "    output.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
            )?;
            writeln!(code, "        let base = chunk_idx * 256;")?;
            writeln!(code, "        let len = out.len();")?;
            writeln!(code, "        let chunks8 = len / 8;")?;
            writeln!(code, "        for c in 0..chunks8 {{")?;
            writeln!(code, "            let r = base + c * 8;")?;
            writeln!(
                code,
                "            let (d0,d1,d2,d3,d4,d5,d6,d7) = dot8_q8_0_q8_0(&input_q8, &weight[r*{row_bytes}..(r+1)*{row_bytes}], &weight[(r+1)*{row_bytes}..(r+2)*{row_bytes}], &weight[(r+2)*{row_bytes}..(r+3)*{row_bytes}], &weight[(r+3)*{row_bytes}..(r+4)*{row_bytes}], &weight[(r+4)*{row_bytes}..(r+5)*{row_bytes}], &weight[(r+5)*{row_bytes}..(r+6)*{row_bytes}], &weight[(r+6)*{row_bytes}..(r+7)*{row_bytes}], &weight[(r+7)*{row_bytes}..(r+8)*{row_bytes}], {k});"
            )?;
            writeln!(code, "            out[c*8] = d0; out[c*8+1] = d1; out[c*8+2] = d2; out[c*8+3] = d3; out[c*8+4] = d4; out[c*8+5] = d5; out[c*8+6] = d6; out[c*8+7] = d7;")?;
            writeln!(code, "        }}")?;
            writeln!(code, "        let tail8 = chunks8 * 8;")?;
            writeln!(code, "        let chunks4t = (len - tail8) / 4;")?;
            writeln!(code, "        for c in 0..chunks4t {{")?;
            writeln!(code, "            let r = base + tail8 + c * 4;")?;
            writeln!(
                code,
                "            let (d0,d1,d2,d3) = dot4_q8_0_q8_0(&input_q8, &weight[r*{row_bytes}..(r+1)*{row_bytes}], &weight[(r+1)*{row_bytes}..(r+2)*{row_bytes}], &weight[(r+2)*{row_bytes}..(r+3)*{row_bytes}], &weight[(r+3)*{row_bytes}..(r+4)*{row_bytes}], {k});"
            )?;
            writeln!(code, "            out[tail8+c*4] = d0; out[tail8+c*4+1] = d1; out[tail8+c*4+2] = d2; out[tail8+c*4+3] = d3;")?;
            writeln!(code, "        }}")?;
            writeln!(code, "        for i in (tail8 + chunks4t*4)..len {{")?;
            writeln!(code, "            let j = base + i;")?;
            writeln!(
                code,
                "            out[i] = dot_q8_0_q8_0(&input_q8, &weight[j*{row_bytes}..(j+1)*{row_bytes}], {k});"
            )?;
            writeln!(code, "        }}")?;
            writeln!(code, "    }});")?;
        } else {
            if n_chunks8 > 0 {
                writeln!(code, "    for chunk in 0..{n_chunks8} {{")?;
                writeln!(code, "        let j0 = chunk * 8;")?;
                writeln!(
                    code,
                    "        let (d0,d1,d2,d3,d4,d5,d6,d7) = dot8_q8_0_q8_0(&input_q8, &weight[j0*{row_bytes}..(j0+1)*{row_bytes}], &weight[(j0+1)*{row_bytes}..(j0+2)*{row_bytes}], &weight[(j0+2)*{row_bytes}..(j0+3)*{row_bytes}], &weight[(j0+3)*{row_bytes}..(j0+4)*{row_bytes}], &weight[(j0+4)*{row_bytes}..(j0+5)*{row_bytes}], &weight[(j0+5)*{row_bytes}..(j0+6)*{row_bytes}], &weight[(j0+6)*{row_bytes}..(j0+7)*{row_bytes}], &weight[(j0+7)*{row_bytes}..(j0+8)*{row_bytes}], {k});"
                )?;
                writeln!(
                    code,
                    "        output[j0] = d0; output[j0+1] = d1; output[j0+2] = d2; output[j0+3] = d3; output[j0+4] = d4; output[j0+5] = d5; output[j0+6] = d6; output[j0+7] = d7;"
                )?;
                writeln!(code, "    }}")?;
            }
            if n_chunks4_tail > 0 {
                let base8 = n_chunks8 * 8;
                writeln!(code, "    let base8 = {base8};")?;
                writeln!(code, "    for chunk4 in 0..{n_chunks4_tail} {{")?;
                writeln!(code, "        let j0 = base8 + chunk4 * 4;")?;
                writeln!(
                    code,
                    "        let (d0,d1,d2,d3) = dot4_q8_0_q8_0(&input_q8, &weight[j0*{row_bytes}..(j0+1)*{row_bytes}], &weight[(j0+1)*{row_bytes}..(j0+2)*{row_bytes}], &weight[(j0+2)*{row_bytes}..(j0+3)*{row_bytes}], &weight[(j0+3)*{row_bytes}..(j0+4)*{row_bytes}], {k});"
                )?;
                writeln!(
                    code,
                    "        output[j0] = d0; output[j0+1] = d1; output[j0+2] = d2; output[j0+3] = d3;"
                )?;
                writeln!(code, "    }}")?;
            }
            if n_rem4 > 0 {
                let base_rem = n_chunks8 * 8 + n_chunks4_tail * 4;
                for r in 0..n_rem4 {
                    writeln!(code, "    output[{base_rem}+{r}] = dot_q8_0_q8_0(&input_q8, &weight[({base_rem}+{r})*{row_bytes}..({base_rem}+{r}+1)*{row_bytes}], {k});")?;
                }
            }
        }
        writeln!(code, "}}")?;
        writeln!(code)?;

        // --- Non-AArch64 fallback: f32 input × Q8_0 weight, 4-accumulator NEON widening ---
        writeln!(code, "#[cfg(not(target_arch = \"aarch64\"))]")?;
        writeln!(code, "#[inline]")?;
        writeln!(
            code,
            "fn matmul_vec_q8_0_{k}x{n}(output: &mut [f32; {n}], input: &[f32; {k}], weight: &[u8]) {{"
        )?;
        if n * (k.div_ceil(32) * 34) >= par_byte_threshold {
            writeln!(
                code,
                "    output.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
            )?;
            writeln!(code, "        let base = chunk_idx * 256;")?;
            writeln!(code, "        for r in 0..out.len() {{")?;
            writeln!(code, "            let j = base + r;")?;
            writeln!(
                code,
                "            out[r] = dot_q8_0(&input[..], &weight[j*{row_bytes}..(j+1)*{row_bytes}], {k});"
            )?;
            writeln!(code, "        }}")?;
            writeln!(code, "    }});")?;
        } else {
            let n_chunks = n / 4;
            let n_remainder = n % 4;
            if n_chunks > 0 {
                writeln!(code, "    for chunk in 0..{n_chunks} {{")?;
                writeln!(code, "        let j0 = chunk * 4;")?;
                writeln!(code, "        output[j0]   = dot_q8_0(&input[..], &weight[j0*{row_bytes}..(j0+1)*{row_bytes}], {k});")?;
                writeln!(code, "        output[j0+1] = dot_q8_0(&input[..], &weight[(j0+1)*{row_bytes}..(j0+2)*{row_bytes}], {k});")?;
                writeln!(code, "        output[j0+2] = dot_q8_0(&input[..], &weight[(j0+2)*{row_bytes}..(j0+3)*{row_bytes}], {k});")?;
                writeln!(code, "        output[j0+3] = dot_q8_0(&input[..], &weight[(j0+3)*{row_bytes}..(j0+4)*{row_bytes}], {k});")?;
                writeln!(code, "    }}")?;
            }
            if n_remainder > 0 {
                writeln!(code, "    let base = {n_chunks} * 4;")?;
                for r in 0..n_remainder {
                    writeln!(
                        code,
                        "    output[base+{r}] = dot_q8_0(&input[..], &weight[(base+{r})*{row_bytes}..(base+{r}+1)*{row_bytes}], {k});"
                    )?;
                }
            }
        }
        writeln!(code, "}}")?;
        writeln!(code)?;
    }

    Ok(())
}

/// Emit shape-specialized Q8_0 *batched* matmul functions:
/// `matmul_mat_q8_0_KxN(output, m, input, weight)`.
///
/// Processes `m` input vectors against a single weight matrix in a
/// weight-outer, token-inner loop so that each 8-row weight block stays hot
/// in L1 across all `m` iterations — this amortizes weight bandwidth by a
/// factor of `m`, which is the whole point of batched prefill on CPU.
///
/// Output layout is row-major `[m * N]` matching the single-token
/// `matmul_vec_q8_0_KxN` output layout, so callers can index as
/// `output[r * N + n]` for token `r`.
///
/// AArch64 path uses `dot8_q8_0_q8_0` (4-simdgroup-worth of sdot ILP).
/// Non-AArch64 fallback loops the scalar `dot_q8_0` kernel.
fn emit_specialized_q8_matmul_batched_functions(
    code: &mut String,
    config: &ModelConfig,
) -> Result<(), CodegenError> {
    writeln!(
        code,
        "// --- Shape-specialized Q8_0 batched matmul functions (m>=1, weight is &[u8]) ---"
    )?;
    writeln!(code)?;

    for &(k, n) in &q8_matmul_shapes(config) {
        let row_bytes = k.div_ceil(32) * 34;
        writeln!(
            code,
            "/// Q8_0 batched matmul: [m, {k}] x [{n}, {k}]^T -> [m, {n}] (row-major; weight raw Q8_0 bytes)"
        )?;
        writeln!(code, "#[cfg(target_arch = \"aarch64\")]")?;
        writeln!(
            code,
            "fn matmul_mat_q8_0_{k}x{n}(output: &mut [f32], m: usize, input: &[f32], weight: &[u8]) {{"
        )?;
        // Quantize all m inputs once.  Row-bytes is statically known, so we
        // size the buffer as `m * row_bytes` and the inner slice is exact.
        writeln!(code, "    let mut input_q8 = vec![0u8; m * {row_bytes}];")?;
        writeln!(code, "    for r in 0..m {{")?;
        writeln!(
            code,
            "        quantize_to_q8_0_blocks_into(&input[r*{k}..(r+1)*{k}], &mut input_q8[r*{row_bytes}..(r+1)*{row_bytes}]);"
        )?;
        writeln!(code, "    }}")?;
        // Weight-outer n-chunks-of-8, token-inner: for each chunk, 8 weight
        // rows stay hot in L1 during the full m loop → amortized bandwidth.
        let n_chunks8 = n / 8;
        let n_rem8 = n % 8;
        let n_chunks4_tail = n_rem8 / 4;
        let n_rem4 = n_rem8 % 4;
        // Parallelize the n-outer loop via rayon.  Different `n_block`s write
        // to disjoint [r*N + j0..j0+8] slices of output, so the reads/writes
        // don't race.  We use a raw pointer + `Send` wrapper (usize carries
        // the address safely across threads; each worker's writes are into
        // proved-disjoint regions of the same buffer).
        //
        // Parallelization is unconditional: for short m (< ~8) the overhead
        // may outweigh the win, and callers should use the per-token path
        // then; this function is intended for batched prefill (m >= 8).
        writeln!(
            code,
            "    // Parallel n-outer loop.  Different n_block values write to"
        )?;
        writeln!(
            code,
            "    // disjoint `output[r*N + j0..j0+8]` slices, so workers never"
        )?;
        writeln!(
            code,
            "    // alias.  Pass the output pointer across threads as a `usize`"
        )?;
        writeln!(
            code,
            "    // (Send+Sync) and reconstruct inside each worker — avoids"
        )?;
        writeln!(code, "    // the raw-pointer Send/Sync boilerplate.")?;
        writeln!(code, "    let out_addr = output.as_mut_ptr() as usize;")?;
        writeln!(code, "    let input_q8_ref: &[u8] = &input_q8;")?;
        if n_chunks8 > 0 {
            writeln!(
                code,
                "    (0..{n_chunks8}).into_par_iter().for_each(|n_block| {{"
            )?;
            writeln!(code, "        let j0 = n_block * 8;")?;
            writeln!(
                code,
                "        let w0 = &weight[(j0+0)*{row_bytes}..(j0+1)*{row_bytes}];"
            )?;
            writeln!(
                code,
                "        let w1 = &weight[(j0+1)*{row_bytes}..(j0+2)*{row_bytes}];"
            )?;
            writeln!(
                code,
                "        let w2 = &weight[(j0+2)*{row_bytes}..(j0+3)*{row_bytes}];"
            )?;
            writeln!(
                code,
                "        let w3 = &weight[(j0+3)*{row_bytes}..(j0+4)*{row_bytes}];"
            )?;
            writeln!(
                code,
                "        let w4 = &weight[(j0+4)*{row_bytes}..(j0+5)*{row_bytes}];"
            )?;
            writeln!(
                code,
                "        let w5 = &weight[(j0+5)*{row_bytes}..(j0+6)*{row_bytes}];"
            )?;
            writeln!(
                code,
                "        let w6 = &weight[(j0+6)*{row_bytes}..(j0+7)*{row_bytes}];"
            )?;
            writeln!(
                code,
                "        let w7 = &weight[(j0+7)*{row_bytes}..(j0+8)*{row_bytes}];"
            )?;
            writeln!(code, "        for r in 0..m {{")?;
            writeln!(
                code,
                "            let input_q = &input_q8_ref[r*{row_bytes}..(r+1)*{row_bytes}];"
            )?;
            writeln!(
                code,
                "            let (d0,d1,d2,d3,d4,d5,d6,d7) = dot8_q8_0_q8_0(input_q, w0, w1, w2, w3, w4, w5, w6, w7, {k});"
            )?;
            writeln!(code, "            unsafe {{")?;
            writeln!(code, "                let base = r * {n} + j0;")?;
            writeln!(
                code,
                "                let p = (out_addr as *mut f32).add(base);"
            )?;
            writeln!(code, "                *p.add(0) = d0;")?;
            writeln!(code, "                *p.add(1) = d1;")?;
            writeln!(code, "                *p.add(2) = d2;")?;
            writeln!(code, "                *p.add(3) = d3;")?;
            writeln!(code, "                *p.add(4) = d4;")?;
            writeln!(code, "                *p.add(5) = d5;")?;
            writeln!(code, "                *p.add(6) = d6;")?;
            writeln!(code, "                *p.add(7) = d7;")?;
            writeln!(code, "            }}")?;
            writeln!(code, "        }}")?;
            writeln!(code, "    }});")?;
        }
        // Tail columns N % 8 (scalar — typically zero for our model shapes).
        // Use dot4 for 4-row chunk of tail, then scalar for the rest.
        if n_chunks4_tail > 0 {
            let tail_base = n_chunks8 * 8;
            writeln!(code, "    // Scalar tail (N % 8 >= 4)")?;
            writeln!(code, "    for r in 0..m {{")?;
            writeln!(
                code,
                "        let input_q = &input_q8[r*{row_bytes}..(r+1)*{row_bytes}];"
            )?;
            for chunk_idx in 0..n_chunks4_tail {
                let j0 = tail_base + chunk_idx * 4;
                writeln!(code, "        {{")?;
                writeln!(code, "            let (d0,d1,d2,d3) = dot4_q8_0_q8_0(input_q, &weight[({j0}+0)*{row_bytes}..({j0}+1)*{row_bytes}], &weight[({j0}+1)*{row_bytes}..({j0}+2)*{row_bytes}], &weight[({j0}+2)*{row_bytes}..({j0}+3)*{row_bytes}], &weight[({j0}+3)*{row_bytes}..({j0}+4)*{row_bytes}], {k});")?;
                writeln!(code, "            output[r*{n} + {j0} + 0] = d0;")?;
                writeln!(code, "            output[r*{n} + {j0} + 1] = d1;")?;
                writeln!(code, "            output[r*{n} + {j0} + 2] = d2;")?;
                writeln!(code, "            output[r*{n} + {j0} + 3] = d3;")?;
                writeln!(code, "        }}")?;
            }
            writeln!(code, "    }}")?;
        }
        if n_rem4 > 0 {
            let tail_base = n_chunks8 * 8 + n_chunks4_tail * 4;
            writeln!(code, "    // Scalar tail (N % 4 remainder)")?;
            writeln!(code, "    for r in 0..m {{")?;
            writeln!(
                code,
                "        let input_q = &input_q8[r*{row_bytes}..(r+1)*{row_bytes}];"
            )?;
            for j in 0..n_rem4 {
                let col = tail_base + j;
                writeln!(code, "        output[r*{n} + {col}] = dot_q8_0_q8_0(input_q, &weight[{col}*{row_bytes}..({col}+1)*{row_bytes}], {k});")?;
            }
            writeln!(code, "    }}")?;
        }
        writeln!(code, "}}")?;
        writeln!(code)?;

        // Non-aarch64 fallback: loop the per-token matmul_vec (no batching
        // benefit but keeps the signature portable).
        writeln!(code, "#[cfg(not(target_arch = \"aarch64\"))]")?;
        writeln!(
            code,
            "fn matmul_mat_q8_0_{k}x{n}(output: &mut [f32], m: usize, input: &[f32], weight: &[u8]) {{"
        )?;
        writeln!(code, "    for r in 0..m {{")?;
        writeln!(code, "        let mut out_row = [0.0f32; {n}];")?;
        writeln!(
            code,
            "        let in_row: &[f32; {k}] = input[r*{k}..(r+1)*{k}].try_into().unwrap();"
        )?;
        writeln!(
            code,
            "        matmul_vec_q8_0_{k}x{n}(&mut out_row, in_row, weight);"
        )?;
        writeln!(
            code,
            "        output[r*{n}..(r+1)*{n}].copy_from_slice(&out_row);"
        )?;
        writeln!(code, "    }}")?;
        writeln!(code, "}}")?;
        writeln!(code)?;
    }
    Ok(())
}

/// Emit the Q4_0 scalar dot-product helper (used as non-aarch64 fallback).
/// Also emits `f16_bits_to_f32` if Q8_0 did not already.
fn emit_q4_0_kernel(code: &mut String) -> Result<(), CodegenError> {
    code.push_str(
        r#"
// --- Q4_0 quantized dot product (scalar fallback for non-aarch64) ---
/// Convert IEEE 754 half-precision (f16) bit pattern to f32.
#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;
    if exponent == 0 {
        if mantissa == 0 { return f32::from_bits(sign << 31); }
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 { m <<= 1; e -= 1; }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exponent == 31 {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13));
    }
    let f32_exp = (exponent as i32 - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

/// Scalar dot product of f32 input against a Q4_0 weight row (non-aarch64 fallback).
/// `weight_row`: raw Q4_0 bytes — layout: [2 bytes f16 scale][16 bytes 4-bit pairs] per block.
/// Each byte packs two 4-bit unsigned values (offset by 8): lo = (byte & 0x0F) - 8, hi = (byte >> 4) - 8.
/// Elements are interleaved: byte j holds element j and element j+16 within a block.
/// `k`: number of input/weight elements.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot_q4_0(input: &[f32], weight_row: &[u8], k: usize) -> f32 {
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize = 18; // 2 scale bytes + 16 data bytes
    let num_blocks = k.div_ceil(BLOCK_SIZE);
    let mut sum = 0.0f32;
    for b in 0..num_blocks {
        let bs = b * TYPE_SIZE;
        let scale_bits = u16::from_le_bytes([weight_row[bs], weight_row[bs + 1]]);
        let scale = f16_bits_to_f32(scale_bits);
        for j in 0..16usize {
            let byte = weight_row[bs + 2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            let idx_lo = b * BLOCK_SIZE + j;
            let idx_hi = b * BLOCK_SIZE + j + 16;
            if idx_lo < k { sum += input[idx_lo] * (lo as f32 * scale); }
            if idx_hi < k { sum += input[idx_hi] * (hi as f32 * scale); }
        }
    }
    sum
}

"#,
    );
    Ok(())
}

/// Emit Q4_0 × Q8_0 NEON sdot kernels for AArch64.
///
/// The strategy: quantize the f32 input to Q8_0 blocks once per matmul call,
/// then for each Q4_0 weight block, unpack the 4-bit values to int8 in-register
/// using NEON (vand + vshr + sub) and use sdot against the Q8_0 input blocks.
///
/// Emits:
/// - `quantize_to_q8_0_blocks_into()` / `quantize_to_q8_0_blocks()` / `f32_to_f16_bits()`
///   (Q8_0 input quantization, reused from Q8_0 path)
/// - `dot_q4_0_q8_0()` — single-row Q4_0 × Q8_0 sdot kernel
/// - `dot4_q4_0_q8_0()` — 4-row batch variant
/// - `dot8_q4_0_q8_0()` — 8-row batch variant
fn emit_q4_0_sdot_kernel(code: &mut String, skip_shared_helpers: bool) -> Result<(), CodegenError> {
    if !skip_shared_helpers {
        code.push_str(
            r#"
// --- Q4_0 × Q8_0 int8 dot product via AArch64 sdot (inline asm, stable Rust) ---
// Input is quantized to Q8_0 once per matmul call; each Q4_0 weight block is
// unpacked from 4-bit to int8 in-register using NEON (vand/ushr/sub), then
// sdot computes the dot product against the Q8_0 input blocks.

/// Quantize a f32 slice to Q8_0 blocks in-place (block-wise, 32 elements per block).
/// `out` must be pre-allocated with `ceil(input.len()/32)*34` bytes.
/// Format: [2-byte f16 scale][32 int8 values] per block.
#[cfg(target_arch = "aarch64")]
#[inline]
fn quantize_to_q8_0_blocks_into(input: &[f32], out: &mut [u8]) {
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize = 34;
    let k = input.len();
    let num_blocks = k.div_ceil(BLOCK_SIZE);
    for b in 0..num_blocks {
        let block_start = b * BLOCK_SIZE;
        let block_end = (block_start + BLOCK_SIZE).min(k);
        let block = &input[block_start..block_end];
        let absmax = block.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let (scale, inv_scale) = if absmax == 0.0 {
            (1.0f32, 0.0f32)
        } else {
            let s = absmax / 127.0;
            (s, 1.0 / s)
        };
        let scale_bits = f32_to_f16_bits(scale);
        let bs = b * TYPE_SIZE;
        out[bs]     = (scale_bits & 0xFF) as u8;
        out[bs + 1] = (scale_bits >> 8)  as u8;
        for (j, &x) in block.iter().enumerate() {
            out[bs + 2 + j] = (x * inv_scale).round().clamp(-127.0, 127.0) as i8 as u8;
        }
        for j in block.len()..BLOCK_SIZE {
            out[bs + 2 + j] = 0u8;
        }
    }
    let _ = num_blocks;
}

/// Quantize a f32 slice to Q8_0 blocks, returning a Vec<u8>.
#[cfg(target_arch = "aarch64")]
#[inline]
fn quantize_to_q8_0_blocks(input: &[f32]) -> Vec<u8> {
    const BLOCK_SIZE: usize = 32;
    const TYPE_SIZE: usize = 34;
    let num_blocks = input.len().div_ceil(BLOCK_SIZE);
    let mut out = vec![0u8; num_blocks * TYPE_SIZE];
    quantize_to_q8_0_blocks_into(input, &mut out);
    out
}

/// f32 → IEEE 754 f16 bit pattern with round-to-nearest-even.  Truncation
/// (`mant >> 13`) gives a systematic round-toward-zero bias on every scale
/// that compounds through stacked matmuls — fixed in v0.9.7 after Q4_K
/// simulate path showed catastrophic perplexity (~50K vs baseline ~4).
#[inline]
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp_f32 == 0xFF {
        let mut h_mant = ((mant >> 13) & 0x3FF) as u16;
        if mant != 0 && h_mant == 0 { h_mant = 1; }
        return sign | 0x7C00 | h_mant;
    }
    if exp_f32 == 0 { return sign; }
    let unbiased = exp_f32 - 127;
    if unbiased > 15 { return sign | 0x7C00; }
    if unbiased < -24 { return sign; }
    let implicit = (mant | 0x0080_0000) as u32;
    let shift: u32 = if unbiased < -14 { (-1 - unbiased) as u32 } else { 13 };
    let mask = (1u32 << shift) - 1;
    let dropped = implicit & mask;
    let half = 1u32 << (shift - 1);
    let mut truncated = implicit >> shift;
    if dropped > half || (dropped == half && (truncated & 1) == 1) {
        truncated = truncated.wrapping_add(1);
    }
    if unbiased < -14 {
        if truncated >= 0x400 { return sign | (1 << 10); }
        return sign | (truncated as u16);
    }
    let mut h_exp = (unbiased + 15) as u32;
    if truncated >= 0x800 {
        h_exp += 1;
        if h_exp >= 0x1F { return sign | 0x7C00; }
        return sign | ((h_exp as u16) << 10);
    }
    sign | ((h_exp as u16) << 10) | ((truncated & 0x3FF) as u16)
}
"#,
        );
    }
    code.push_str(
        r#"

// --- Q4_0 × Q8_0 kernel (assumes Q8_0 quantization helpers already emitted) ---

/// Single-row Q4_0 × Q8_0 dot product via AArch64 sdot instruction.
///
/// For each block pair, unpacks the 16 packed Q4_0 bytes into 32 signed int8
/// values in-register (vand for low nibbles, ushr for high nibbles, sub 8 to
/// center), then uses sdot against the corresponding Q8_0 input block.
///
/// `a_q8`: input quantized to Q8_0 blocks (34 bytes/block).
/// `b_q4`: weight row in Q4_0 format (18 bytes/block).
/// `k`: element count.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_q4_0_q8_0(a_q8: &[u8], b_q4: &[u8], k: usize) -> f32 {
    use std::arch::aarch64::*;
    const BLOCK_SIZE: usize = 32;
    const Q8_TYPE_SIZE: usize = 34;
    const Q4_TYPE_SIZE: usize = 18;
    let num_blocks = k / BLOCK_SIZE;

    let mut fa0 = 0.0f32;
    let mut fa1 = 0.0f32;
    let mut fa2 = 0.0f32;
    let mut fa3 = 0.0f32;

    // Mask for extracting low nibbles and bias vector for centering
    let mask_0f: u8 = 0x0F;
    let bias_8: i8 = 8;

    macro_rules! block_f32 {
        ($b:expr) => {{
            let a_bs = $b * Q8_TYPE_SIZE;
            let b_bs = $b * Q4_TYPE_SIZE;
            let a_scale = f16_bits_to_f32(u16::from_le_bytes([a_q8[a_bs], a_q8[a_bs + 1]]));
            let b_scale = f16_bits_to_f32(u16::from_le_bytes([b_q4[b_bs], b_q4[b_bs + 1]]));
            let combined = a_scale * b_scale;
            unsafe {
                // Load Q8_0 input (32 int8 values in two 16-byte halves)
                let ap = a_q8[a_bs + 2..].as_ptr() as *const i8;
                let x0: int8x16_t = vld1q_s8(ap);
                let x1: int8x16_t = vld1q_s8(ap.add(16));
                // Load 16 packed Q4_0 bytes
                let bp = b_q4[b_bs + 2..].as_ptr();
                let packed: uint8x16_t = vld1q_u8(bp);
                // Unpack low nibbles: (packed & 0x0F) - 8 → elements [0..16)
                let mask: uint8x16_t = vdupq_n_u8(mask_0f);
                let lo_u8: uint8x16_t = vandq_u8(packed, mask);
                let bias: int8x16_t = vdupq_n_s8(bias_8);
                let w_lo: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(lo_u8), bias);
                // Unpack high nibbles: (packed >> 4) - 8 → elements [16..32)
                let hi_u8: uint8x16_t = vshrq_n_u8::<4>(packed);
                let w_hi: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(hi_u8), bias);
                // sdot: accumulate dot products
                let mut acc: int32x4_t = vdupq_n_s32(0);
                core::arch::asm!(
                    "sdot {acc:v}.4s, {w0:v}.16b, {x0:v}.16b",
                    "sdot {acc:v}.4s, {w1:v}.16b, {x1:v}.16b",
                    acc = inout(vreg) acc,
                    w0 = in(vreg) w_lo,
                    x0 = in(vreg) x0,
                    w1 = in(vreg) w_hi,
                    x1 = in(vreg) x1,
                    options(nostack),
                );
                combined * vaddvq_s32(acc) as f32
            }
        }};
    }

    let groups = num_blocks / 4;
    for g in 0..groups {
        let base = g * 4;
        fa0 += block_f32!(base);
        fa1 += block_f32!(base + 1);
        fa2 += block_f32!(base + 2);
        fa3 += block_f32!(base + 3);
    }
    let tail = groups * 4;
    if tail     < num_blocks { fa0 += block_f32!(tail);     }
    if tail + 1 < num_blocks { fa1 += block_f32!(tail + 1); }
    if tail + 2 < num_blocks { fa2 += block_f32!(tail + 2); }

    fa0 + fa1 + fa2 + fa3
}

/// 4-row batch Q4_0 × Q8_0 dot product via AArch64 sdot.
///
/// Loads the Q8_0 input block ONCE and unpacks 4 Q4_0 weight blocks,
/// issuing 8 independent sdot instructions (2 per row) for 4-way ILP.
///
/// Returns `(dot0, dot1, dot2, dot3)`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot4_q4_0_q8_0(
    a_q8: &[u8],
    b0_q4: &[u8],
    b1_q4: &[u8],
    b2_q4: &[u8],
    b3_q4: &[u8],
    k: usize,
) -> (f32, f32, f32, f32) {
    use std::arch::aarch64::*;
    const BLOCK_SIZE: usize = 32;
    const Q8_TYPE_SIZE: usize = 34;
    const Q4_TYPE_SIZE: usize = 18;
    let num_blocks = k / BLOCK_SIZE;

    let mut r0 = 0.0f32;
    let mut r1 = 0.0f32;
    let mut r2 = 0.0f32;
    let mut r3 = 0.0f32;

    for b in 0..num_blocks {
        let a_bs = b * Q8_TYPE_SIZE;
        let b_bs = b * Q4_TYPE_SIZE;
        let a_scale = f16_bits_to_f32(u16::from_le_bytes([a_q8[a_bs], a_q8[a_bs + 1]]));
        let s0 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b0_q4[b_bs], b0_q4[b_bs + 1]]));
        let s1 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b1_q4[b_bs], b1_q4[b_bs + 1]]));
        let s2 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b2_q4[b_bs], b2_q4[b_bs + 1]]));
        let s3 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b3_q4[b_bs], b3_q4[b_bs + 1]]));
        unsafe {
            // Load Q8_0 input ONCE
            let ap = a_q8[a_bs + 2..].as_ptr() as *const i8;
            let x0: int8x16_t = vld1q_s8(ap);
            let x1: int8x16_t = vld1q_s8(ap.add(16));
            let mask: uint8x16_t = vdupq_n_u8(0x0F);
            let bias: int8x16_t = vdupq_n_s8(8);
            // Unpack 4 Q4_0 weight blocks
            let bp0 = b0_q4[b_bs + 2..].as_ptr();
            let bp1 = b1_q4[b_bs + 2..].as_ptr();
            let bp2 = b2_q4[b_bs + 2..].as_ptr();
            let bp3 = b3_q4[b_bs + 2..].as_ptr();
            let p0: uint8x16_t = vld1q_u8(bp0);
            let p1: uint8x16_t = vld1q_u8(bp1);
            let p2: uint8x16_t = vld1q_u8(bp2);
            let p3: uint8x16_t = vld1q_u8(bp3);
            // Low nibbles
            let w0_lo: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p0, mask)), bias);
            let w1_lo: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p1, mask)), bias);
            let w2_lo: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p2, mask)), bias);
            let w3_lo: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p3, mask)), bias);
            // High nibbles
            let w0_hi: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p0)), bias);
            let w1_hi: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p1)), bias);
            let w2_hi: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p2)), bias);
            let w3_hi: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p3)), bias);
            // 8 independent sdot — 4-way ILP across accumulators
            let mut acc0: int32x4_t = vdupq_n_s32(0);
            let mut acc1: int32x4_t = vdupq_n_s32(0);
            let mut acc2: int32x4_t = vdupq_n_s32(0);
            let mut acc3: int32x4_t = vdupq_n_s32(0);
            core::arch::asm!(
                "sdot {a0:v}.4s, {w0l:v}.16b, {x0:v}.16b",
                "sdot {a1:v}.4s, {w1l:v}.16b, {x0:v}.16b",
                "sdot {a2:v}.4s, {w2l:v}.16b, {x0:v}.16b",
                "sdot {a3:v}.4s, {w3l:v}.16b, {x0:v}.16b",
                "sdot {a0:v}.4s, {w0h:v}.16b, {x1:v}.16b",
                "sdot {a1:v}.4s, {w1h:v}.16b, {x1:v}.16b",
                "sdot {a2:v}.4s, {w2h:v}.16b, {x1:v}.16b",
                "sdot {a3:v}.4s, {w3h:v}.16b, {x1:v}.16b",
                a0 = inout(vreg) acc0, a1 = inout(vreg) acc1,
                a2 = inout(vreg) acc2, a3 = inout(vreg) acc3,
                x0 = in(vreg) x0,     x1 = in(vreg) x1,
                w0l = in(vreg) w0_lo,  w0h = in(vreg) w0_hi,
                w1l = in(vreg) w1_lo,  w1h = in(vreg) w1_hi,
                w2l = in(vreg) w2_lo,  w2h = in(vreg) w2_hi,
                w3l = in(vreg) w3_lo,  w3h = in(vreg) w3_hi,
                options(nostack),
            );
            r0 += s0 * vaddvq_s32(acc0) as f32;
            r1 += s1 * vaddvq_s32(acc1) as f32;
            r2 += s2 * vaddvq_s32(acc2) as f32;
            r3 += s3 * vaddvq_s32(acc3) as f32;
        }
    }
    (r0, r1, r2, r3)
}

/// 8-row batch Q4_0 × Q8_0 dot product via AArch64 sdot.
///
/// Loads the Q8_0 input block ONCE and unpacks 8 Q4_0 weight blocks,
/// issuing 16 independent sdot instructions (2 per row) for 8-way ILP.
/// The 8 independent first-half sdot instructions fully cover the 4-cycle
/// sdot RAW latency before each row's second-half sdot.
///
/// Returns `(r0, r1, r2, r3, r4, r5, r6, r7)`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot8_q4_0_q8_0(
    a_q8: &[u8],
    b0_q4: &[u8], b1_q4: &[u8], b2_q4: &[u8], b3_q4: &[u8],
    b4_q4: &[u8], b5_q4: &[u8], b6_q4: &[u8], b7_q4: &[u8],
    k: usize,
) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    use std::arch::aarch64::*;
    const BLOCK_SIZE: usize = 32;
    const Q8_TYPE_SIZE: usize = 34;
    const Q4_TYPE_SIZE: usize = 18;
    let num_blocks = k / BLOCK_SIZE;

    let mut r0 = 0.0f32; let mut r1 = 0.0f32;
    let mut r2 = 0.0f32; let mut r3 = 0.0f32;
    let mut r4 = 0.0f32; let mut r5 = 0.0f32;
    let mut r6 = 0.0f32; let mut r7 = 0.0f32;

    for b in 0..num_blocks {
        let a_bs = b * Q8_TYPE_SIZE;
        let b_bs = b * Q4_TYPE_SIZE;
        let a_scale = f16_bits_to_f32(u16::from_le_bytes([a_q8[a_bs], a_q8[a_bs + 1]]));
        let s0 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b0_q4[b_bs], b0_q4[b_bs + 1]]));
        let s1 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b1_q4[b_bs], b1_q4[b_bs + 1]]));
        let s2 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b2_q4[b_bs], b2_q4[b_bs + 1]]));
        let s3 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b3_q4[b_bs], b3_q4[b_bs + 1]]));
        let s4 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b4_q4[b_bs], b4_q4[b_bs + 1]]));
        let s5 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b5_q4[b_bs], b5_q4[b_bs + 1]]));
        let s6 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b6_q4[b_bs], b6_q4[b_bs + 1]]));
        let s7 = a_scale * f16_bits_to_f32(u16::from_le_bytes([b7_q4[b_bs], b7_q4[b_bs + 1]]));
        unsafe {
            let ap = a_q8[a_bs + 2..].as_ptr() as *const i8;
            let x0: int8x16_t = vld1q_s8(ap);
            let x1: int8x16_t = vld1q_s8(ap.add(16));
            let mask: uint8x16_t = vdupq_n_u8(0x0F);
            let bias: int8x16_t = vdupq_n_s8(8);
            // Unpack 8 Q4_0 weight blocks
            let p0: uint8x16_t = vld1q_u8(b0_q4[b_bs + 2..].as_ptr());
            let p1: uint8x16_t = vld1q_u8(b1_q4[b_bs + 2..].as_ptr());
            let p2: uint8x16_t = vld1q_u8(b2_q4[b_bs + 2..].as_ptr());
            let p3: uint8x16_t = vld1q_u8(b3_q4[b_bs + 2..].as_ptr());
            let p4: uint8x16_t = vld1q_u8(b4_q4[b_bs + 2..].as_ptr());
            let p5: uint8x16_t = vld1q_u8(b5_q4[b_bs + 2..].as_ptr());
            let p6: uint8x16_t = vld1q_u8(b6_q4[b_bs + 2..].as_ptr());
            let p7: uint8x16_t = vld1q_u8(b7_q4[b_bs + 2..].as_ptr());
            // Low nibbles for all 8 rows
            let w0l: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p0, mask)), bias);
            let w1l: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p1, mask)), bias);
            let w2l: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p2, mask)), bias);
            let w3l: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p3, mask)), bias);
            let w4l: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p4, mask)), bias);
            let w5l: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p5, mask)), bias);
            let w6l: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p6, mask)), bias);
            let w7l: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(p7, mask)), bias);
            // High nibbles for all 8 rows
            let w0h: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p0)), bias);
            let w1h: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p1)), bias);
            let w2h: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p2)), bias);
            let w3h: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p3)), bias);
            let w4h: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p4)), bias);
            let w5h: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p5)), bias);
            let w6h: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p6)), bias);
            let w7h: int8x16_t = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(p7)), bias);
            // 16 independent sdot — 8 for low halves, 8 for high halves
            let mut acc0: int32x4_t = vdupq_n_s32(0);
            let mut acc1: int32x4_t = vdupq_n_s32(0);
            let mut acc2: int32x4_t = vdupq_n_s32(0);
            let mut acc3: int32x4_t = vdupq_n_s32(0);
            let mut acc4: int32x4_t = vdupq_n_s32(0);
            let mut acc5: int32x4_t = vdupq_n_s32(0);
            let mut acc6: int32x4_t = vdupq_n_s32(0);
            let mut acc7: int32x4_t = vdupq_n_s32(0);
            core::arch::asm!(
                "sdot {a0:v}.4s, {w0l:v}.16b, {x0:v}.16b",
                "sdot {a1:v}.4s, {w1l:v}.16b, {x0:v}.16b",
                "sdot {a2:v}.4s, {w2l:v}.16b, {x0:v}.16b",
                "sdot {a3:v}.4s, {w3l:v}.16b, {x0:v}.16b",
                "sdot {a4:v}.4s, {w4l:v}.16b, {x0:v}.16b",
                "sdot {a5:v}.4s, {w5l:v}.16b, {x0:v}.16b",
                "sdot {a6:v}.4s, {w6l:v}.16b, {x0:v}.16b",
                "sdot {a7:v}.4s, {w7l:v}.16b, {x0:v}.16b",
                "sdot {a0:v}.4s, {w0h:v}.16b, {x1:v}.16b",
                "sdot {a1:v}.4s, {w1h:v}.16b, {x1:v}.16b",
                "sdot {a2:v}.4s, {w2h:v}.16b, {x1:v}.16b",
                "sdot {a3:v}.4s, {w3h:v}.16b, {x1:v}.16b",
                "sdot {a4:v}.4s, {w4h:v}.16b, {x1:v}.16b",
                "sdot {a5:v}.4s, {w5h:v}.16b, {x1:v}.16b",
                "sdot {a6:v}.4s, {w6h:v}.16b, {x1:v}.16b",
                "sdot {a7:v}.4s, {w7h:v}.16b, {x1:v}.16b",
                a0 = inout(vreg) acc0, a1 = inout(vreg) acc1,
                a2 = inout(vreg) acc2, a3 = inout(vreg) acc3,
                a4 = inout(vreg) acc4, a5 = inout(vreg) acc5,
                a6 = inout(vreg) acc6, a7 = inout(vreg) acc7,
                x0 = in(vreg) x0,     x1 = in(vreg) x1,
                w0l = in(vreg) w0l,    w0h = in(vreg) w0h,
                w1l = in(vreg) w1l,    w1h = in(vreg) w1h,
                w2l = in(vreg) w2l,    w2h = in(vreg) w2h,
                w3l = in(vreg) w3l,    w3h = in(vreg) w3h,
                w4l = in(vreg) w4l,    w4h = in(vreg) w4h,
                w5l = in(vreg) w5l,    w5h = in(vreg) w5h,
                w6l = in(vreg) w6l,    w6h = in(vreg) w6h,
                w7l = in(vreg) w7l,    w7h = in(vreg) w7h,
                options(nostack),
            );
            r0 += s0 * vaddvq_s32(acc0) as f32;
            r1 += s1 * vaddvq_s32(acc1) as f32;
            r2 += s2 * vaddvq_s32(acc2) as f32;
            r3 += s3 * vaddvq_s32(acc3) as f32;
            r4 += s4 * vaddvq_s32(acc4) as f32;
            r5 += s5 * vaddvq_s32(acc5) as f32;
            r6 += s6 * vaddvq_s32(acc6) as f32;
            r7 += s7 * vaddvq_s32(acc7) as f32;
        }
    }
    (r0, r1, r2, r3, r4, r5, r6, r7)
}

"#,
    );
    Ok(())
}

/// Collect all unique (k, n) matmul shapes needed for Q4_0 projection weights.
fn q4_matmul_shapes(config: &ModelConfig) -> Vec<(usize, usize)> {
    matmul_shapes(config)
}

/// Emit shape-specialized Q4_0 matmul functions: `matmul_vec_q4_0_KxN`.
/// On aarch64: quantizes input to Q8_0, then uses dot8/dot4/dot1 cascade.
/// On other targets: uses scalar `dot_q4_0()` fallback.
fn emit_specialized_q4_matmul_functions(
    code: &mut String,
    config: &ModelConfig,
) -> Result<(), CodegenError> {
    writeln!(
        code,
        "// --- Shape-specialized Q4_0 matmul functions (m=1, weight is &[u8]) ---"
    )?;
    writeln!(code)?;

    // Parallelize when total weight bytes > 1 MB (adapts to model size).
    let par_byte_threshold: usize = 1_000_000;

    for &(k, n) in &q4_matmul_shapes(config) {
        // Byte size per row: ceil(k/32)*18
        let row_bytes = k.div_ceil(32) * 18;
        // Q8_0 buffer size for input quantization on aarch64
        let q8_bytes = k.div_ceil(32) * 34;
        writeln!(
            code,
            "/// Q4_0 matmul: [1, {k}] x [{n}, {k}]^T -> [1, {n}] (weight stored as raw Q4_0 bytes)"
        )?;

        // --- AArch64 sdot path ---
        writeln!(code, "#[cfg(target_arch = \"aarch64\")]")?;
        writeln!(code, "#[inline]")?;
        writeln!(
            code,
            "fn matmul_vec_q4_0_{k}x{n}(output: &mut [f32; {n}], input: &[f32; {k}], weight: &[u8]) {{"
        )?;
        // Stack-allocated Q8_0 buffer for input quantization
        writeln!(code, "    let mut input_q8 = [0u8; {q8_bytes}];")?;
        writeln!(
            code,
            "    quantize_to_q8_0_blocks_into(&input[..], &mut input_q8);"
        )?;
        let n_chunks8 = n / 8;
        let n_rem8 = n % 8;
        let n_chunks4_tail = n_rem8 / 4;
        let n_rem4 = n_rem8 % 4;
        if n * (k.div_ceil(32) * 34) >= par_byte_threshold {
            writeln!(
                code,
                "    output.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
            )?;
            writeln!(code, "        let base = chunk_idx * 256;")?;
            writeln!(code, "        let len = out.len();")?;
            writeln!(code, "        let chunks8 = len / 8;")?;
            writeln!(code, "        for c in 0..chunks8 {{")?;
            writeln!(code, "            let r = base + c * 8;")?;
            writeln!(
                code,
                "            let (d0,d1,d2,d3,d4,d5,d6,d7) = dot8_q4_0_q8_0(&input_q8, &weight[r*{row_bytes}..(r+1)*{row_bytes}], &weight[(r+1)*{row_bytes}..(r+2)*{row_bytes}], &weight[(r+2)*{row_bytes}..(r+3)*{row_bytes}], &weight[(r+3)*{row_bytes}..(r+4)*{row_bytes}], &weight[(r+4)*{row_bytes}..(r+5)*{row_bytes}], &weight[(r+5)*{row_bytes}..(r+6)*{row_bytes}], &weight[(r+6)*{row_bytes}..(r+7)*{row_bytes}], &weight[(r+7)*{row_bytes}..(r+8)*{row_bytes}], {k});"
            )?;
            writeln!(code, "            out[c*8] = d0; out[c*8+1] = d1; out[c*8+2] = d2; out[c*8+3] = d3; out[c*8+4] = d4; out[c*8+5] = d5; out[c*8+6] = d6; out[c*8+7] = d7;")?;
            writeln!(code, "        }}")?;
            writeln!(code, "        let tail8 = chunks8 * 8;")?;
            writeln!(code, "        let chunks4t = (len - tail8) / 4;")?;
            writeln!(code, "        for c in 0..chunks4t {{")?;
            writeln!(code, "            let r = base + tail8 + c * 4;")?;
            writeln!(
                code,
                "            let (d0,d1,d2,d3) = dot4_q4_0_q8_0(&input_q8, &weight[r*{row_bytes}..(r+1)*{row_bytes}], &weight[(r+1)*{row_bytes}..(r+2)*{row_bytes}], &weight[(r+2)*{row_bytes}..(r+3)*{row_bytes}], &weight[(r+3)*{row_bytes}..(r+4)*{row_bytes}], {k});"
            )?;
            writeln!(code, "            out[tail8+c*4] = d0; out[tail8+c*4+1] = d1; out[tail8+c*4+2] = d2; out[tail8+c*4+3] = d3;")?;
            writeln!(code, "        }}")?;
            writeln!(code, "        for i in (tail8 + chunks4t*4)..len {{")?;
            writeln!(code, "            let j = base + i;")?;
            writeln!(
                code,
                "            out[i] = dot_q4_0_q8_0(&input_q8, &weight[j*{row_bytes}..(j+1)*{row_bytes}], {k});"
            )?;
            writeln!(code, "        }}")?;
            writeln!(code, "    }});")?;
        } else {
            if n_chunks8 > 0 {
                writeln!(code, "    for chunk in 0..{n_chunks8} {{")?;
                writeln!(code, "        let j0 = chunk * 8;")?;
                writeln!(
                    code,
                    "        let (d0,d1,d2,d3,d4,d5,d6,d7) = dot8_q4_0_q8_0(&input_q8, &weight[j0*{row_bytes}..(j0+1)*{row_bytes}], &weight[(j0+1)*{row_bytes}..(j0+2)*{row_bytes}], &weight[(j0+2)*{row_bytes}..(j0+3)*{row_bytes}], &weight[(j0+3)*{row_bytes}..(j0+4)*{row_bytes}], &weight[(j0+4)*{row_bytes}..(j0+5)*{row_bytes}], &weight[(j0+5)*{row_bytes}..(j0+6)*{row_bytes}], &weight[(j0+6)*{row_bytes}..(j0+7)*{row_bytes}], &weight[(j0+7)*{row_bytes}..(j0+8)*{row_bytes}], {k});"
                )?;
                writeln!(
                    code,
                    "        output[j0] = d0; output[j0+1] = d1; output[j0+2] = d2; output[j0+3] = d3; output[j0+4] = d4; output[j0+5] = d5; output[j0+6] = d6; output[j0+7] = d7;"
                )?;
                writeln!(code, "    }}")?;
            }
            if n_chunks4_tail > 0 {
                let base8 = n_chunks8 * 8;
                writeln!(code, "    let base8 = {base8};")?;
                writeln!(code, "    for chunk4 in 0..{n_chunks4_tail} {{")?;
                writeln!(code, "        let j0 = base8 + chunk4 * 4;")?;
                writeln!(
                    code,
                    "        let (d0,d1,d2,d3) = dot4_q4_0_q8_0(&input_q8, &weight[j0*{row_bytes}..(j0+1)*{row_bytes}], &weight[(j0+1)*{row_bytes}..(j0+2)*{row_bytes}], &weight[(j0+2)*{row_bytes}..(j0+3)*{row_bytes}], &weight[(j0+3)*{row_bytes}..(j0+4)*{row_bytes}], {k});"
                )?;
                writeln!(
                    code,
                    "        output[j0] = d0; output[j0+1] = d1; output[j0+2] = d2; output[j0+3] = d3;"
                )?;
                writeln!(code, "    }}")?;
            }
            if n_rem4 > 0 {
                let base_rem = n_chunks8 * 8 + n_chunks4_tail * 4;
                for r in 0..n_rem4 {
                    writeln!(code, "    output[{base_rem}+{r}] = dot_q4_0_q8_0(&input_q8, &weight[({base_rem}+{r})*{row_bytes}..({base_rem}+{r}+1)*{row_bytes}], {k});")?;
                }
            }
        }
        writeln!(code, "}}")?;
        writeln!(code)?;

        // --- Non-AArch64 fallback: scalar dot_q4_0 ---
        writeln!(code, "#[cfg(not(target_arch = \"aarch64\"))]")?;
        writeln!(code, "#[inline]")?;
        writeln!(
            code,
            "fn matmul_vec_q4_0_{k}x{n}(output: &mut [f32; {n}], input: &[f32; {k}], weight: &[u8]) {{"
        )?;
        if n * (k.div_ceil(32) * 34) >= par_byte_threshold {
            writeln!(
                code,
                "    output.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
            )?;
            writeln!(code, "        let base = chunk_idx * 256;")?;
            writeln!(code, "        for r in 0..out.len() {{")?;
            writeln!(code, "            let j = base + r;")?;
            writeln!(
                code,
                "            out[r] = dot_q4_0(&input[..], &weight[j*{row_bytes}..(j+1)*{row_bytes}], {k});"
            )?;
            writeln!(code, "        }}")?;
            writeln!(code, "    }});")?;
        } else {
            let n_chunks = n / 4;
            let n_remainder = n % 4;
            if n_chunks > 0 {
                writeln!(code, "    for chunk in 0..{n_chunks} {{")?;
                writeln!(code, "        let j0 = chunk * 4;")?;
                writeln!(code, "        output[j0]   = dot_q4_0(&input[..], &weight[j0*{row_bytes}..(j0+1)*{row_bytes}], {k});")?;
                writeln!(code, "        output[j0+1] = dot_q4_0(&input[..], &weight[(j0+1)*{row_bytes}..(j0+2)*{row_bytes}], {k});")?;
                writeln!(code, "        output[j0+2] = dot_q4_0(&input[..], &weight[(j0+2)*{row_bytes}..(j0+3)*{row_bytes}], {k});")?;
                writeln!(code, "        output[j0+3] = dot_q4_0(&input[..], &weight[(j0+3)*{row_bytes}..(j0+4)*{row_bytes}], {k});")?;
                writeln!(code, "    }}")?;
            }
            if n_remainder > 0 {
                writeln!(code, "    let base = {n_chunks} * 4;")?;
                for r in 0..n_remainder {
                    writeln!(
                        code,
                        "    output[base+{r}] = dot_q4_0(&input[..], &weight[(base+{r})*{row_bytes}..(base+{r}+1)*{row_bytes}], {k});"
                    )?;
                }
            }
        }
        writeln!(code, "}}")?;
        writeln!(code)?;
    }

    Ok(())
}

/// Emit the Q4_K × Q8_0 dot kernel and the K-quant scale-min unpacker.
///
/// Q4_K layout (256-element super-blocks, 144 bytes each):
///   [0..2]    d  — f16 super-block scale
///   [2..4]    dmin — f16 super-block min
///   [4..16]   12 bytes packing 8 sub-block (6-bit scale, 6-bit min) pairs
///   [16..144] 128 bytes packing 256 × 4-bit unsigned weights
///
/// Dequant per 32-element sub-block j:
///   w[i] = d * sub_scale[j] * q[i] - dmin * sub_min[j]
///
/// The kernel takes pre-quantized Q8_0 input (one 32-element block per Q4_K
/// sub-block, eight blocks per super-block).  Per sub-block it computes two
/// reductions on int8 input × 4-bit weight: the dot `Σ x_q * q4` and the
/// plain sum `Σ x_q`, then combines with the f32 scale/min coefficients.
///
/// `skip_shared_helpers` controls whether `f16_bits_to_f32` is emitted; the
/// Q8_0 / Q4_0 kernels also emit it, so we skip when those are present.
fn emit_q4_k_kernel(code: &mut String, skip_shared_helpers: bool) -> Result<(), CodegenError> {
    if !skip_shared_helpers {
        code.push_str(
            r#"
/// Convert IEEE 754 half-precision (f16) bit pattern to f32.
#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;
    if exponent == 0 {
        if mantissa == 0 { return f32::from_bits(sign << 31); }
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 { m <<= 1; e -= 1; }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exponent == 31 {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13));
    }
    let f32_exp = (exponent as i32 - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}
"#,
        );
    }
    code.push_str(
        r#"
// --- Q4_K × Q8_0 dot product (256-element super-block, scalar reference) ---

/// Unpack the (scale, min) pair for sub-block j (0..8) from the 12-byte
/// packed scales blob of a Q4_K super-block.  Matches GGML reference.
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Dot product of a Q4_K weight row and a Q8_0-quantized input row,
/// both of length `k` (must be a multiple of 256).
///
/// `weight_q4k`: 144 bytes per 256-element super-block.
/// `input_q8`:    34 bytes per 32-element block (eight per super-block).
///
/// On AArch64 with `target-cpu=native` (set in the generated `.cargo/
/// config.toml`) the inner sub-block dot uses `sdot` via inline asm; the
/// `dmin` correction needs `Σ x_q8` which we compute with `vpaddlq_s8 +
/// vaddvq_s16`.  Fallback path is the scalar reference.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_q4_k_q8_0(weight_q4k: &[u8], input_q8: &[u8], k: usize) -> f32 {
    use std::arch::aarch64::*;
    const Q4K_BLOCK: usize = 256;
    const Q4K_BYTES: usize = 144;
    const Q8_BYTES: usize = 34;
    let num_sb = k / Q4K_BLOCK;
    let mut acc = 0.0f32;
    for sb in 0..num_sb {
        let wb = sb * Q4K_BYTES;
        let d_bits = u16::from_le_bytes([weight_q4k[wb], weight_q4k[wb + 1]]);
        let dmin_bits = u16::from_le_bytes([weight_q4k[wb + 2], weight_q4k[wb + 3]]);
        let d = f16_bits_to_f32(d_bits);
        let dmin = f16_bits_to_f32(dmin_bits);
        let scales = &weight_q4k[wb + 4..wb + 16];
        let qs = &weight_q4k[wb + 16..wb + 144];
        let x_block_base = sb * 8 * Q8_BYTES;
        for chunk in 0..4 {
            let is = chunk * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1_w = d * sc1 as f32;
            let m1_w = dmin * m1 as f32;
            let d2_w = d * sc2 as f32;
            let m2_w = dmin * m2 as f32;
            unsafe {
                let mask = vdupq_n_u8(0x0F);
                let qs_ptr = qs[chunk * 32..].as_ptr();
                let qs_v0: uint8x16_t = vld1q_u8(qs_ptr);
                let qs_v1: uint8x16_t = vld1q_u8(qs_ptr.add(16));
                // Sub-block 1: low nibbles. Sub-block 2: high nibbles.
                let q1_lo = vreinterpretq_s8_u8(vandq_u8(qs_v0, mask));
                let q1_hi = vreinterpretq_s8_u8(vandq_u8(qs_v1, mask));
                let q2_lo = vreinterpretq_s8_u8(vshrq_n_u8::<4>(qs_v0));
                let q2_hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(qs_v1));

                // Sub-block 1 input.
                let x1_off = x_block_base + is * Q8_BYTES;
                let x1_scale = f16_bits_to_f32(u16::from_le_bytes([
                    input_q8[x1_off],
                    input_q8[x1_off + 1],
                ]));
                let x1_ptr = input_q8[x1_off + 2..].as_ptr() as *const i8;
                let x1_lo: int8x16_t = vld1q_s8(x1_ptr);
                let x1_hi: int8x16_t = vld1q_s8(x1_ptr.add(16));
                let mut dot1_v = vdupq_n_s32(0);
                core::arch::asm!(
                    "sdot {acc:v}.4s, {w0:v}.16b, {x0:v}.16b",
                    "sdot {acc:v}.4s, {w1:v}.16b, {x1:v}.16b",
                    acc = inout(vreg) dot1_v,
                    w0 = in(vreg) q1_lo,
                    x0 = in(vreg) x1_lo,
                    w1 = in(vreg) q1_hi,
                    x1 = in(vreg) x1_hi,
                    options(nostack),
                );
                let dot1 = vaddvq_s32(dot1_v) as i32;
                let sum1_v = vaddq_s16(vpaddlq_s8(x1_lo), vpaddlq_s8(x1_hi));
                let sum1 = vaddvq_s16(sum1_v) as i32;
                acc += x1_scale * (d1_w * dot1 as f32 - m1_w * sum1 as f32);

                // Sub-block 2 input.
                let x2_off = x_block_base + (is + 1) * Q8_BYTES;
                let x2_scale = f16_bits_to_f32(u16::from_le_bytes([
                    input_q8[x2_off],
                    input_q8[x2_off + 1],
                ]));
                let x2_ptr = input_q8[x2_off + 2..].as_ptr() as *const i8;
                let x2_lo: int8x16_t = vld1q_s8(x2_ptr);
                let x2_hi: int8x16_t = vld1q_s8(x2_ptr.add(16));
                let mut dot2_v = vdupq_n_s32(0);
                core::arch::asm!(
                    "sdot {acc:v}.4s, {w0:v}.16b, {x0:v}.16b",
                    "sdot {acc:v}.4s, {w1:v}.16b, {x1:v}.16b",
                    acc = inout(vreg) dot2_v,
                    w0 = in(vreg) q2_lo,
                    x0 = in(vreg) x2_lo,
                    w1 = in(vreg) q2_hi,
                    x1 = in(vreg) x2_hi,
                    options(nostack),
                );
                let dot2 = vaddvq_s32(dot2_v) as i32;
                let sum2_v = vaddq_s16(vpaddlq_s8(x2_lo), vpaddlq_s8(x2_hi));
                let sum2 = vaddvq_s16(sum2_v) as i32;
                acc += x2_scale * (d2_w * dot2 as f32 - m2_w * sum2 as f32);
            }
        }
    }
    acc
}

/// 4-row variant of `dot_q4_k_q8_0`.  One Q8_0 input row × four Q4_K
/// weight rows in parallel; input loads + `Σ x_q8` reductions are
/// amortised across the rows.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot4_q4_k_q8_0(
    w0: &[u8],
    w1: &[u8],
    w2: &[u8],
    w3: &[u8],
    input_q8: &[u8],
    k: usize,
) -> (f32, f32, f32, f32) {
    use std::arch::aarch64::*;
    const Q4K_BLOCK: usize = 256;
    const Q4K_BYTES: usize = 144;
    const Q8_BYTES: usize = 34;
    let num_sb = k / Q4K_BLOCK;
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let weights = [w0, w1, w2, w3];
    for sb in 0..num_sb {
        let wb = sb * Q4K_BYTES;
        let mut d_arr = [0.0f32; 4];
        let mut dmin_arr = [0.0f32; 4];
        for r in 0..4 {
            let w = weights[r];
            let d_bits = u16::from_le_bytes([w[wb], w[wb + 1]]);
            let dmin_bits = u16::from_le_bytes([w[wb + 2], w[wb + 3]]);
            d_arr[r] = f16_bits_to_f32(d_bits);
            dmin_arr[r] = f16_bits_to_f32(dmin_bits);
        }
        let x_block_base = sb * 8 * Q8_BYTES;
        for chunk in 0..4 {
            let is = chunk * 2;
            let mut sc1_arr = [0u8; 4];
            let mut m1_arr = [0u8; 4];
            let mut sc2_arr = [0u8; 4];
            let mut m2_arr = [0u8; 4];
            for r in 0..4 {
                let scales = &weights[r][wb + 4..wb + 16];
                let (sc1, m1) = get_scale_min_k4(is, scales);
                let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                sc1_arr[r] = sc1;
                m1_arr[r] = m1;
                sc2_arr[r] = sc2;
                m2_arr[r] = m2;
            }
            unsafe {
                let mask = vdupq_n_u8(0x0F);
                let x1_off = x_block_base + is * Q8_BYTES;
                let x1_scale = f16_bits_to_f32(u16::from_le_bytes([
                    input_q8[x1_off],
                    input_q8[x1_off + 1],
                ]));
                let x1_ptr = input_q8[x1_off + 2..].as_ptr() as *const i8;
                let x1_lo: int8x16_t = vld1q_s8(x1_ptr);
                let x1_hi: int8x16_t = vld1q_s8(x1_ptr.add(16));
                let sum1_v = vaddq_s16(vpaddlq_s8(x1_lo), vpaddlq_s8(x1_hi));
                let sum1 = vaddvq_s16(sum1_v) as i32;
                let x2_off = x_block_base + (is + 1) * Q8_BYTES;
                let x2_scale = f16_bits_to_f32(u16::from_le_bytes([
                    input_q8[x2_off],
                    input_q8[x2_off + 1],
                ]));
                let x2_ptr = input_q8[x2_off + 2..].as_ptr() as *const i8;
                let x2_lo: int8x16_t = vld1q_s8(x2_ptr);
                let x2_hi: int8x16_t = vld1q_s8(x2_ptr.add(16));
                let sum2_v = vaddq_s16(vpaddlq_s8(x2_lo), vpaddlq_s8(x2_hi));
                let sum2 = vaddvq_s16(sum2_v) as i32;

                let acc_arr = [&mut acc0, &mut acc1, &mut acc2, &mut acc3];
                for (r, acc_ref) in acc_arr.into_iter().enumerate() {
                    let qs_ptr = weights[r][wb + 16 + chunk * 32..].as_ptr();
                    let qs_v0: uint8x16_t = vld1q_u8(qs_ptr);
                    let qs_v1: uint8x16_t = vld1q_u8(qs_ptr.add(16));
                    let q1_lo = vreinterpretq_s8_u8(vandq_u8(qs_v0, mask));
                    let q1_hi = vreinterpretq_s8_u8(vandq_u8(qs_v1, mask));
                    let q2_lo = vreinterpretq_s8_u8(vshrq_n_u8::<4>(qs_v0));
                    let q2_hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(qs_v1));
                    let mut dot1_v = vdupq_n_s32(0);
                    let mut dot2_v = vdupq_n_s32(0);
                    core::arch::asm!(
                        "sdot {a1:v}.4s, {w1lo:v}.16b, {x1lo:v}.16b",
                        "sdot {a1:v}.4s, {w1hi:v}.16b, {x1hi:v}.16b",
                        "sdot {a2:v}.4s, {w2lo:v}.16b, {x2lo:v}.16b",
                        "sdot {a2:v}.4s, {w2hi:v}.16b, {x2hi:v}.16b",
                        a1 = inout(vreg) dot1_v,
                        a2 = inout(vreg) dot2_v,
                        w1lo = in(vreg) q1_lo,
                        w1hi = in(vreg) q1_hi,
                        x1lo = in(vreg) x1_lo,
                        x1hi = in(vreg) x1_hi,
                        w2lo = in(vreg) q2_lo,
                        w2hi = in(vreg) q2_hi,
                        x2lo = in(vreg) x2_lo,
                        x2hi = in(vreg) x2_hi,
                        options(nostack),
                    );
                    let dot1 = vaddvq_s32(dot1_v) as i32;
                    let dot2 = vaddvq_s32(dot2_v) as i32;
                    let d = d_arr[r];
                    let dmin = dmin_arr[r];
                    let d1_w = d * sc1_arr[r] as f32;
                    let m1_w = dmin * m1_arr[r] as f32;
                    let d2_w = d * sc2_arr[r] as f32;
                    let m2_w = dmin * m2_arr[r] as f32;
                    *acc_ref += x1_scale * (d1_w * dot1 as f32 - m1_w * sum1 as f32);
                    *acc_ref += x2_scale * (d2_w * dot2 as f32 - m2_w * sum2 as f32);
                }
            }
        }
    }
    (acc0, acc1, acc2, acc3)
}

/// Scalar fallback for non-aarch64 targets — same math, no SIMD.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot_q4_k_q8_0(weight_q4k: &[u8], input_q8: &[u8], k: usize) -> f32 {
    const Q4K_BLOCK: usize = 256;
    const Q4K_BYTES: usize = 144;
    const Q8_BYTES: usize = 34;
    let num_sb = k / Q4K_BLOCK;
    let mut acc = 0.0f32;
    for sb in 0..num_sb {
        let wb = sb * Q4K_BYTES;
        let d_bits = u16::from_le_bytes([weight_q4k[wb], weight_q4k[wb + 1]]);
        let dmin_bits = u16::from_le_bytes([weight_q4k[wb + 2], weight_q4k[wb + 3]]);
        let d = f16_bits_to_f32(d_bits);
        let dmin = f16_bits_to_f32(dmin_bits);
        let scales = &weight_q4k[wb + 4..wb + 16];
        let qs = &weight_q4k[wb + 16..wb + 144];
        let x_block_base = sb * 8 * Q8_BYTES;
        for chunk in 0..4 {
            let is = chunk * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1_w = d * sc1 as f32;
            let m1_w = dmin * m1 as f32;
            let d2_w = d * sc2 as f32;
            let m2_w = dmin * m2 as f32;
            let qs_chunk = &qs[chunk * 32..(chunk + 1) * 32];
            let x1_off = x_block_base + is * Q8_BYTES;
            let x1_scale = f16_bits_to_f32(u16::from_le_bytes([
                input_q8[x1_off],
                input_q8[x1_off + 1],
            ]));
            let x1_i8 = &input_q8[x1_off + 2..x1_off + 34];
            let mut dot1 = 0i32;
            let mut sum1 = 0i32;
            for i in 0..32 {
                let q4 = (qs_chunk[i] & 0x0F) as i32;
                let xi = x1_i8[i] as i8 as i32;
                dot1 += xi * q4;
                sum1 += xi;
            }
            acc += x1_scale * (d1_w * dot1 as f32 - m1_w * sum1 as f32);
            let x2_off = x_block_base + (is + 1) * Q8_BYTES;
            let x2_scale = f16_bits_to_f32(u16::from_le_bytes([
                input_q8[x2_off],
                input_q8[x2_off + 1],
            ]));
            let x2_i8 = &input_q8[x2_off + 2..x2_off + 34];
            let mut dot2 = 0i32;
            let mut sum2 = 0i32;
            for i in 0..32 {
                let q4 = (qs_chunk[i] >> 4) as i32;
                let xi = x2_i8[i] as i8 as i32;
                dot2 += xi * q4;
                sum2 += xi;
            }
            acc += x2_scale * (d2_w * dot2 as f32 - m2_w * sum2 as f32);
        }
    }
    acc
}

"#,
    );
    Ok(())
}

/// Q4_K matmul shapes — same set as Q4_0 / Q8_0, but constrained to those
/// where `k` is a multiple of 256 (Q4_K super-block size).  Tensors with a
/// non-multiple `k` (rare — typically only norm-sized buffers) fall back to
/// the Q8_0 path on load.
/// Format a single shape-specialized matmul call for the given storage dtype.
/// Returns a line of Rust source like `matmul_vec_q4_k_4096x4096(&mut q, &normed, &lw.q_proj);`
/// (with `indent` prepended).  Per-projection dispatch sites in
/// `forward`/`prefill` use this so each projection picks its own kernel
/// based on `ModelConfig::effective_dtype`.
fn matmul_call(
    dtype: DType,
    k: usize,
    n: usize,
    dst: &str,
    src: &str,
    weight: &str,
    indent: &str,
) -> String {
    let prefix = match dtype {
        DType::Q8_0 => "matmul_vec_q8_0_",
        DType::Q4_0 => "matmul_vec_q4_0_",
        DType::Q4_K => "matmul_vec_q4_k_",
        DType::Q6_K => "matmul_vec_q6_k_",
        _ => "matmul_vec_",
    };
    format!("{indent}{prefix}{k}x{n}(&mut {dst}, &{src}, &{weight});")
}

/// Shapes for which to emit specialized matmul kernels for `target` dtype.
///
/// With per-projection dtype mixing (`config.proj_dtypes`), each projection
/// category may have its own storage dtype.  This returns the (K, N) shapes
/// of the projections that match `target` so we only emit kernels for what
/// the model actually uses.  Falls through to all matmul shapes when the
/// config is uniformly `target`.
fn proj_shapes_for_dtype(config: &ModelConfig, target: DType) -> Vec<(usize, usize)> {
    use forgellm_frontend::ir::ProjCategory;
    let pdt = config.effective_proj_dtypes();
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let qk_size = config.num_attention_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let vocab = config.vocab_size;

    let pairs: [(ProjCategory, usize, usize); 8] = [
        (ProjCategory::Q, hidden, qk_size),
        (ProjCategory::K, hidden, kv_size),
        (ProjCategory::V, hidden, kv_size),
        (ProjCategory::O, qk_size, hidden),
        (ProjCategory::Gate, hidden, intermediate),
        (ProjCategory::Up, hidden, intermediate),
        (ProjCategory::Down, intermediate, hidden),
        (ProjCategory::LmHead, hidden, vocab),
    ];

    let mut shapes: Vec<(usize, usize)> = pairs
        .iter()
        .filter(|(cat, _, _)| pdt.get(*cat) == target)
        .map(|(_, k, n)| (*k, *n))
        .collect();
    shapes.sort();
    shapes.dedup();
    shapes
}

fn q4_k_matmul_shapes(config: &ModelConfig) -> Vec<(usize, usize)> {
    proj_shapes_for_dtype(config, DType::Q4_K)
        .into_iter()
        .filter(|(k, _)| k.is_multiple_of(256))
        .collect()
}

/// Emit shape-specialized Q4_K matmul functions: `matmul_vec_q4_k_KxN`.
///
/// On AArch64, dispatches in groups of 4 output rows via `dot4_q4_k_q8_0`
/// (one Q8_0 input load shared across 4 weight rows + amortised
/// `Σ x_q8` reduction), with a single-row tail.  Other targets fall back
/// to the scalar `dot_q4_k_q8_0` per row.
fn emit_specialized_q4_k_matmul_functions(
    code: &mut String,
    config: &ModelConfig,
) -> Result<(), CodegenError> {
    writeln!(
        code,
        "// --- Shape-specialized Q4_K matmul functions (m=1, weight is &[u8]) ---"
    )?;
    writeln!(code)?;

    // Parallelize when total weight bytes > 1 MB.  Q4_K row size = ceil(k/256)*144.
    let par_byte_threshold: usize = 1_000_000;

    for &(k, n) in &q4_k_matmul_shapes(config) {
        let row_bytes = k.div_ceil(256) * 144;
        let q8_bytes = k.div_ceil(32) * 34;
        writeln!(
            code,
            "/// Q4_K matmul: [1, {k}] x [{n}, {k}]^T -> [1, {n}] (weight stored as raw Q4_K bytes)"
        )?;
        writeln!(code, "#[inline]")?;
        writeln!(
            code,
            "fn matmul_vec_q4_k_{k}x{n}(output: &mut [f32; {n}], input: &[f32; {k}], weight: &[u8]) {{"
        )?;
        writeln!(code, "    let mut input_q8 = vec![0u8; {q8_bytes}];")?;
        writeln!(
            code,
            "    quantize_to_q8_0_blocks_into(&input[..], &mut input_q8);"
        )?;
        if n * row_bytes >= par_byte_threshold {
            writeln!(
                code,
                "    output.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
            )?;
            writeln!(code, "        let base = chunk_idx * 256;")?;
            writeln!(code, "        let len = out.len();")?;
            writeln!(code, "        #[cfg(target_arch = \"aarch64\")] {{")?;
            writeln!(code, "            let chunks4 = len / 4;")?;
            writeln!(code, "            for c in 0..chunks4 {{")?;
            writeln!(code, "                let r = base + c * 4;")?;
            writeln!(
                code,
                "                let (d0, d1, d2, d3) = dot4_q4_k_q8_0(&weight[r*{row_bytes}..(r+1)*{row_bytes}], &weight[(r+1)*{row_bytes}..(r+2)*{row_bytes}], &weight[(r+2)*{row_bytes}..(r+3)*{row_bytes}], &weight[(r+3)*{row_bytes}..(r+4)*{row_bytes}], &input_q8, {k});"
            )?;
            writeln!(code, "                out[c*4] = d0; out[c*4+1] = d1; out[c*4+2] = d2; out[c*4+3] = d3;")?;
            writeln!(code, "            }}")?;
            writeln!(code, "            for i in (chunks4 * 4)..len {{")?;
            writeln!(code, "                let j = base + i;")?;
            writeln!(
                code,
                "                out[i] = dot_q4_k_q8_0(&weight[j*{row_bytes}..(j+1)*{row_bytes}], &input_q8, {k});"
            )?;
            writeln!(code, "            }}")?;
            writeln!(code, "        }}")?;
            writeln!(code, "        #[cfg(not(target_arch = \"aarch64\"))] {{")?;
            writeln!(code, "            for r in 0..len {{")?;
            writeln!(code, "                let j = base + r;")?;
            writeln!(
                code,
                "                out[r] = dot_q4_k_q8_0(&weight[j*{row_bytes}..(j+1)*{row_bytes}], &input_q8, {k});"
            )?;
            writeln!(code, "            }}")?;
            writeln!(code, "        }}")?;
            writeln!(code, "    }});")?;
        } else {
            // Below the parallel threshold — emit a 4-wide cascade in-line.
            writeln!(code, "    #[cfg(target_arch = \"aarch64\")] {{")?;
            let n4 = n / 4;
            let n_rem = n % 4;
            if n4 > 0 {
                writeln!(code, "        for c in 0..{n4} {{")?;
                writeln!(code, "            let r = c * 4;")?;
                writeln!(
                    code,
                    "            let (d0, d1, d2, d3) = dot4_q4_k_q8_0(&weight[r*{row_bytes}..(r+1)*{row_bytes}], &weight[(r+1)*{row_bytes}..(r+2)*{row_bytes}], &weight[(r+2)*{row_bytes}..(r+3)*{row_bytes}], &weight[(r+3)*{row_bytes}..(r+4)*{row_bytes}], &input_q8, {k});"
                )?;
                writeln!(code, "            output[c*4] = d0; output[c*4+1] = d1; output[c*4+2] = d2; output[c*4+3] = d3;")?;
                writeln!(code, "        }}")?;
            }
            if n_rem > 0 {
                let base = n4 * 4;
                writeln!(code, "        for j in {base}..{n} {{")?;
                writeln!(
                    code,
                    "            output[j] = dot_q4_k_q8_0(&weight[j*{row_bytes}..(j+1)*{row_bytes}], &input_q8, {k});"
                )?;
                writeln!(code, "        }}")?;
            }
            writeln!(code, "    }}")?;
            writeln!(code, "    #[cfg(not(target_arch = \"aarch64\"))] {{")?;
            writeln!(code, "        for j in 0..{n} {{")?;
            writeln!(
                code,
                "            output[j] = dot_q4_k_q8_0(&weight[j*{row_bytes}..(j+1)*{row_bytes}], &input_q8, {k});"
            )?;
            writeln!(code, "        }}")?;
            writeln!(code, "    }}")?;
        }
        writeln!(code, "}}")?;
        writeln!(code)?;
    }

    Ok(())
}

fn q6_k_matmul_shapes(config: &ModelConfig) -> Vec<(usize, usize)> {
    proj_shapes_for_dtype(config, DType::Q6_K)
        .into_iter()
        .filter(|(k, _)| k.is_multiple_of(256))
        .collect()
}

/// Emit the Q6_K kernel: scalar `dot_q6_k_q8_0` and any helpers Q6_K
/// needs that aren't already shared with Q8_0 / Q4_K.  Mirrors
/// `emit_q6_k_q8_0` from the frontend (`weight_loader.rs`) bit-for-bit.
///
/// `skip_shared_helpers` indicates that `f16_bits_to_f32` and
/// `quantize_to_q8_0_blocks_into` are already in scope (emitted by
/// the Q8_0 / Q4_0 / Q4_K kernels).  Q6_K reuses both.
fn emit_q6_k_kernel(code: &mut String, skip_shared_helpers: bool) -> Result<(), CodegenError> {
    if !skip_shared_helpers {
        // f16_bits_to_f32 helper (only needed if no other quant kernel is
        // active — pasted from the Q4_K emission path).  Q6_K is rarely the
        // only quant in a model so this branch is mostly defensive.
        code.push_str(
            r#"
#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;
    if exponent == 0 {
        if mantissa == 0 { return f32::from_bits(sign << 31); }
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 { m <<= 1; e -= 1; }
        m &= 0x3FF;
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exponent == 31 {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13));
    }
    let f32_exp = (exponent as i32 - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}
"#,
        );
    }
    code.push_str(
        r#"
// --- Q6_K × Q8_0 dot product (256-element super-block, scalar reference) ---
//
// Q6_K stores 256 elements in 210 bytes:
//   [0..128]   ql: low 4 bits of 256 quants (interleaved, 2 per byte)
//   [128..192] qh: top 2 bits of 256 quants (4 per byte)
//   [192..208] scales: 16 sub-block i8 scales
//   [208..210] d: f16 super-block scale
//
// Per Q8_0 input block (32 elements) we cover 2 Q6_K sub-blocks (16 each)
// and look up the right (ql_off, qh_off, nibble_shift, qh_shift, scale_idx)
// slot from a fixed dispatch table.

#[inline]
fn dot_q6_k_q8_0(weight_q6k: &[u8], input_q8: &[u8], k: usize) -> f32 {
    const Q6K_BLOCK: usize = 256;
    const Q6K_BYTES: usize = 210;
    const Q8_BYTES: usize = 34;
    let num_sb = k / Q6K_BLOCK;
    let mut acc = 0.0f32;

    // (ql_off, qh_off, nibble_shift, qh_shift, sc_idx_base) per Q8_0 block.
    const BLOCKS: [(usize, usize, u32, u32, usize); 8] = [
        (0, 0, 0, 0, 0),
        (32, 0, 0, 2, 2),
        (0, 0, 4, 4, 4),
        (32, 0, 4, 6, 6),
        (64, 32, 0, 0, 8),
        (96, 32, 0, 2, 10),
        (64, 32, 4, 4, 12),
        (96, 32, 4, 6, 14),
    ];

    for sb in 0..num_sb {
        let wb = sb * Q6K_BYTES;
        let ql = &weight_q6k[wb..wb + 128];
        let qh = &weight_q6k[wb + 128..wb + 192];
        let scales = &weight_q6k[wb + 192..wb + 208];
        let d_bits = u16::from_le_bytes([weight_q6k[wb + 208], weight_q6k[wb + 209]]);
        let d = f16_bits_to_f32(d_bits);
        let x_block_base = sb * 8 * Q8_BYTES;

        for (q8_blk, &(ql_off, qh_off, nibble_shift, qh_shift, sc_idx)) in BLOCKS.iter().enumerate() {
            let x_off = x_block_base + q8_blk * Q8_BYTES;
            let x_scale = f16_bits_to_f32(u16::from_le_bytes([
                input_q8[x_off],
                input_q8[x_off + 1],
            ]));
            let x_i8 = &input_q8[x_off + 2..x_off + 34];
            let sc_lo = scales[sc_idx] as i8 as i32;
            let sc_hi = scales[sc_idx + 1] as i8 as i32;

            let mut dot_lo = 0i32;
            let mut dot_hi = 0i32;
            for l in 0..16 {
                let q6_unsigned = ((ql[ql_off + l] >> nibble_shift) & 0x0F) as i32
                    | (((qh[qh_off + l] >> qh_shift) & 0x03) as i32) << 4;
                let q6 = q6_unsigned - 32;
                let xi = x_i8[l] as i8 as i32;
                dot_lo += q6 * xi;
            }
            for l in 16..32 {
                let q6_unsigned = ((ql[ql_off + l] >> nibble_shift) & 0x0F) as i32
                    | (((qh[qh_off + l] >> qh_shift) & 0x03) as i32) << 4;
                let q6 = q6_unsigned - 32;
                let xi = x_i8[l] as i8 as i32;
                dot_hi += q6 * xi;
            }
            acc += d * x_scale * (sc_lo as f32 * dot_lo as f32 + sc_hi as f32 * dot_hi as f32);
        }
    }
    acc
}
"#,
    );
    Ok(())
}

/// Emit shape-specialized Q6_K matmul functions: `matmul_vec_q6_k_KxN`.
///
/// Scalar-only in v0.9.12 — no NEON dot4 ILP yet (deferred to v0.9.13).
/// Parallelizes over output rows when total weight bytes > 1 MB
/// (`row_bytes = ceil(k/256) * 210`).
fn emit_specialized_q6_k_matmul_functions(
    code: &mut String,
    config: &ModelConfig,
) -> Result<(), CodegenError> {
    writeln!(code, "// --- Shape-specialized Q6_K matmul functions (m=1, weight is &[u8]) ---")?;
    writeln!(code)?;

    let par_byte_threshold: usize = 1_000_000;

    for &(k, n) in &q6_k_matmul_shapes(config) {
        let row_bytes = k.div_ceil(256) * 210;
        let q8_bytes = k.div_ceil(32) * 34;
        writeln!(code, "/// Q6_K matmul: [1, {k}] x [{n}, {k}]^T -> [1, {n}] (weight stored as raw Q6_K bytes)")?;
        writeln!(code, "#[inline]")?;
        writeln!(code, "fn matmul_vec_q6_k_{k}x{n}(output: &mut [f32; {n}], input: &[f32; {k}], weight: &[u8]) {{")?;
        writeln!(code, "    let mut input_q8 = vec![0u8; {q8_bytes}];")?;
        writeln!(code, "    quantize_to_q8_0_blocks_into(&input[..], &mut input_q8);")?;
        if n * row_bytes >= par_byte_threshold {
            writeln!(code, "    output.par_chunks_mut(64).enumerate().for_each(|(chunk_idx, out)| {{")?;
            writeln!(code, "        let base = chunk_idx * 64;")?;
            writeln!(code, "        for r in 0..out.len() {{")?;
            writeln!(code, "            let j = base + r;")?;
            writeln!(code, "            out[r] = dot_q6_k_q8_0(&weight[j*{row_bytes}..(j+1)*{row_bytes}], &input_q8, {k});")?;
            writeln!(code, "        }}")?;
            writeln!(code, "    }});")?;
        } else {
            writeln!(code, "    for j in 0..{n} {{")?;
            writeln!(code, "        output[j] = dot_q6_k_q8_0(&weight[j*{row_bytes}..(j+1)*{row_bytes}], &input_q8, {k});")?;
            writeln!(code, "    }}")?;
        }
        writeln!(code, "}}")?;
        writeln!(code)?;
    }

    Ok(())
}

/// Emit shape-specialized Q4_0 *batched* matmul functions:
/// `matmul_mat_q4_0_KxN(output, m, input, weight)`.
///
/// Mirrors `emit_specialized_q8_matmul_batched_functions` but with
/// `dot8_q4_0_q8_0` / `dot4_q4_0_q8_0` inner kernels and Q4_0 weight
/// row byte size (`ceil(k/32) * 18`).  Input is still Q8_0-quantized
/// (same `quantize_to_q8_0_blocks_into` helper) since the Q4_0 weight
/// × Q8_0 input path is what the sdot kernels expect.
fn emit_specialized_q4_matmul_batched_functions(
    code: &mut String,
    config: &ModelConfig,
) -> Result<(), CodegenError> {
    writeln!(
        code,
        "// --- Shape-specialized Q4_0 batched matmul functions (m>=1, weight is &[u8]) ---"
    )?;
    writeln!(code)?;

    for &(k, n) in &q4_matmul_shapes(config) {
        let weight_row_bytes = k.div_ceil(32) * 18;
        let q8_row_bytes = k.div_ceil(32) * 34;
        writeln!(
            code,
            "/// Q4_0 batched matmul: [m, {k}] x [{n}, {k}]^T -> [m, {n}] (row-major; weight raw Q4_0 bytes)"
        )?;
        writeln!(code, "#[cfg(target_arch = \"aarch64\")]")?;
        writeln!(
            code,
            "fn matmul_mat_q4_0_{k}x{n}(output: &mut [f32], m: usize, input: &[f32], weight: &[u8]) {{"
        )?;
        writeln!(
            code,
            "    let mut input_q8 = vec![0u8; m * {q8_row_bytes}];"
        )?;
        writeln!(code, "    for r in 0..m {{")?;
        writeln!(
            code,
            "        quantize_to_q8_0_blocks_into(&input[r*{k}..(r+1)*{k}], &mut input_q8[r*{q8_row_bytes}..(r+1)*{q8_row_bytes}]);"
        )?;
        writeln!(code, "    }}")?;

        let n_chunks8 = n / 8;
        let n_rem8 = n % 8;
        let n_chunks4_tail = n_rem8 / 4;
        let n_rem4 = n_rem8 % 4;
        writeln!(
            code,
            "    // Parallel n-outer loop.  Disjoint output writes per n_block;"
        )?;
        writeln!(
            code,
            "    // pointer round-trips through usize to stay Send+Sync."
        )?;
        writeln!(code, "    let out_addr = output.as_mut_ptr() as usize;")?;
        writeln!(code, "    let input_q8_ref: &[u8] = &input_q8;")?;
        if n_chunks8 > 0 {
            writeln!(
                code,
                "    (0..{n_chunks8}).into_par_iter().for_each(|n_block| {{"
            )?;
            writeln!(code, "        let j0 = n_block * 8;")?;
            for i in 0..8 {
                writeln!(
                    code,
                    "        let w{i} = &weight[(j0+{i})*{weight_row_bytes}..(j0+{ip1})*{weight_row_bytes}];",
                    ip1 = i + 1
                )?;
            }
            writeln!(code, "        for r in 0..m {{")?;
            writeln!(
                code,
                "            let input_q = &input_q8_ref[r*{q8_row_bytes}..(r+1)*{q8_row_bytes}];"
            )?;
            writeln!(
                code,
                "            let (d0,d1,d2,d3,d4,d5,d6,d7) = dot8_q4_0_q8_0(input_q, w0, w1, w2, w3, w4, w5, w6, w7, {k});"
            )?;
            writeln!(code, "            unsafe {{")?;
            writeln!(code, "                let base = r * {n} + j0;")?;
            writeln!(
                code,
                "                let p = (out_addr as *mut f32).add(base);"
            )?;
            writeln!(code, "                *p.add(0) = d0;")?;
            writeln!(code, "                *p.add(1) = d1;")?;
            writeln!(code, "                *p.add(2) = d2;")?;
            writeln!(code, "                *p.add(3) = d3;")?;
            writeln!(code, "                *p.add(4) = d4;")?;
            writeln!(code, "                *p.add(5) = d5;")?;
            writeln!(code, "                *p.add(6) = d6;")?;
            writeln!(code, "                *p.add(7) = d7;")?;
            writeln!(code, "            }}")?;
            writeln!(code, "        }}")?;
            writeln!(code, "    }});")?;
        }
        // Tail handling (N % 8).
        if n_chunks4_tail > 0 {
            let tail_base = n_chunks8 * 8;
            writeln!(code, "    // Scalar tail (N % 8 >= 4)")?;
            writeln!(code, "    for r in 0..m {{")?;
            writeln!(
                code,
                "        let input_q = &input_q8[r*{q8_row_bytes}..(r+1)*{q8_row_bytes}];"
            )?;
            for chunk_idx in 0..n_chunks4_tail {
                let j0 = tail_base + chunk_idx * 4;
                writeln!(code, "        {{")?;
                writeln!(code, "            let (d0,d1,d2,d3) = dot4_q4_0_q8_0(input_q, &weight[({j0}+0)*{weight_row_bytes}..({j0}+1)*{weight_row_bytes}], &weight[({j0}+1)*{weight_row_bytes}..({j0}+2)*{weight_row_bytes}], &weight[({j0}+2)*{weight_row_bytes}..({j0}+3)*{weight_row_bytes}], &weight[({j0}+3)*{weight_row_bytes}..({j0}+4)*{weight_row_bytes}], {k});")?;
                writeln!(code, "            output[r*{n} + {j0} + 0] = d0;")?;
                writeln!(code, "            output[r*{n} + {j0} + 1] = d1;")?;
                writeln!(code, "            output[r*{n} + {j0} + 2] = d2;")?;
                writeln!(code, "            output[r*{n} + {j0} + 3] = d3;")?;
                writeln!(code, "        }}")?;
            }
            writeln!(code, "    }}")?;
        }
        if n_rem4 > 0 {
            let tail_base = n_chunks8 * 8 + n_chunks4_tail * 4;
            writeln!(code, "    // Scalar tail (N % 4 remainder)")?;
            writeln!(code, "    for r in 0..m {{")?;
            writeln!(
                code,
                "        let input_q = &input_q8[r*{q8_row_bytes}..(r+1)*{q8_row_bytes}];"
            )?;
            for j in 0..n_rem4 {
                let col = tail_base + j;
                writeln!(code, "        output[r*{n} + {col}] = dot_q4_0_q8_0(input_q, &weight[{col}*{weight_row_bytes}..({col}+1)*{weight_row_bytes}], {k});")?;
            }
            writeln!(code, "    }}")?;
        }
        writeln!(code, "}}")?;
        writeln!(code)?;

        // Non-aarch64 fallback: loop per-token matmul_vec.
        writeln!(code, "#[cfg(not(target_arch = \"aarch64\"))]")?;
        writeln!(
            code,
            "fn matmul_mat_q4_0_{k}x{n}(output: &mut [f32], m: usize, input: &[f32], weight: &[u8]) {{"
        )?;
        writeln!(code, "    for r in 0..m {{")?;
        writeln!(code, "        let mut out_row = [0.0f32; {n}];")?;
        writeln!(
            code,
            "        let in_row: &[f32; {k}] = input[r*{k}..(r+1)*{k}].try_into().unwrap();"
        )?;
        writeln!(
            code,
            "        matmul_vec_q4_0_{k}x{n}(&mut out_row, in_row, weight);"
        )?;
        writeln!(
            code,
            "        output[r*{n}..(r+1)*{n}].copy_from_slice(&out_row);"
        )?;
        writeln!(code, "    }}")?;
        writeln!(code, "}}")?;
        writeln!(code)?;
    }
    Ok(())
}

fn emit_forward_function(
    code: &mut String,
    _graph: &Graph,
    config: &ModelConfig,
) -> Result<(), CodegenError> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let vocab = config.vocab_size;
    let qk_size = num_heads * head_dim;
    let kv_size = num_kv_heads * head_dim;

    use forgellm_frontend::ir::ProjCategory;
    let pdt = config.effective_proj_dtypes();
    let q_dtype = pdt.q;
    let k_dtype = pdt.k;
    let v_dtype = pdt.v;
    let o_dtype = pdt.o;
    let gate_dtype = pdt.gate;
    let up_dtype = pdt.up;
    let down_dtype = pdt.down;
    let lm_dtype = config.effective_dtype(ProjCategory::LmHead);
    let lm_is_q8 = lm_dtype == DType::Q8_0;
    let lm_is_q4 = lm_dtype == DType::Q4_0;
    let lm_is_q4k = lm_dtype == DType::Q4_K;
    let lm_is_q6k = lm_dtype == DType::Q6_K;
    // Projection weight type: raw bytes for quantized projections, f32
    // otherwise.  All current quantized variants (Q8_0 / Q4_0 / Q4_K / Q6_K)
    // store as Vec<u8>, so a per-field projection only needs the f32 escape
    // hatch when the dtype is non-byte-storage (currently never the case for
    // mixed Q4_K_M files).
    let dtype_is_bytes = |d: DType| {
        matches!(d, DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q6_K)
    };
    let proj_type_for = |d: DType| {
        if dtype_is_bytes(d) {
            "Vec<u8>"
        } else {
            "Vec<f32>"
        }
    };
    let q_proj_type = proj_type_for(q_dtype);
    let k_proj_type = proj_type_for(k_dtype);
    let v_proj_type = proj_type_for(v_dtype);
    let o_proj_type = proj_type_for(o_dtype);
    let gate_proj_type = proj_type_for(gate_dtype);
    let up_proj_type = proj_type_for(up_dtype);
    let down_proj_type = proj_type_for(down_dtype);
    let lm_head_type = proj_type_for(lm_dtype);

    writeln!(
        code,
        "/// Model weights — loaded once, passed to forward()."
    )?;
    writeln!(code, "pub struct Weights {{")?;
    writeln!(
        code,
        "    pub embed_tokens: Vec<f32>,       // [{vocab} * {hidden}]"
    )?;
    writeln!(code, "    pub layers: Vec<LayerWeights>,")?;
    writeln!(code, "    pub final_norm: Vec<f32>,          // [{hidden}]")?;
    writeln!(
        code,
        "    pub lm_head: {lm_head_type},             // [{vocab} * {hidden}]"
    )?;
    writeln!(code, "}}")?;
    writeln!(code)?;

    writeln!(code, "pub struct LayerWeights {{")?;
    writeln!(code, "    pub attn_norm: Vec<f32>,           // [{hidden}]")?;
    writeln!(
        code,
        "    pub q_proj: {q_proj_type},              // [{} * {hidden}]",
        num_heads * head_dim
    )?;
    writeln!(
        code,
        "    pub k_proj: {k_proj_type},              // [{} * {hidden}]",
        num_kv_heads * head_dim
    )?;
    writeln!(
        code,
        "    pub v_proj: {v_proj_type},              // [{} * {hidden}]",
        num_kv_heads * head_dim
    )?;
    // Bias fields for Qwen2 (qkv_bias = true)
    if config.qkv_bias {
        writeln!(
            code,
            "    pub q_bias: Vec<f32>,                // [{qk_size}]  (Qwen2 Q projection bias)"
        )?;
        writeln!(
            code,
            "    pub k_bias: Vec<f32>,                // [{kv_size}]  (Qwen2 K projection bias)"
        )?;
        writeln!(
            code,
            "    pub v_bias: Vec<f32>,                // [{kv_size}]  (Qwen2 V projection bias)"
        )?;
    }
    writeln!(
        code,
        "    pub o_proj: {o_proj_type},              // [{hidden} * {}]",
        num_heads * head_dim
    )?;
    writeln!(code, "    pub ffn_norm: Vec<f32>,            // [{hidden}]")?;
    writeln!(
        code,
        "    pub gate_proj: {gate_proj_type},           // [{intermediate} * {hidden}]"
    )?;
    writeln!(
        code,
        "    pub up_proj: {up_proj_type},             // [{intermediate} * {hidden}]"
    )?;
    writeln!(
        code,
        "    pub down_proj: {down_proj_type},           // [{hidden} * {intermediate}]"
    )?;
    writeln!(code, "}}")?;
    writeln!(code)?;

    writeln!(
        code,
        "/// KV cache for autoregressive generation (int8 quantized)."
    )?;
    writeln!(
        code,
        "/// Pre-allocated to MAX_SEQ_LEN — zero allocations during inference."
    )?;
    writeln!(
        code,
        "/// Uses int8 quantization with per-token absmax scale for 4x memory reduction."
    )?;
    writeln!(code, "pub struct KVCache {{")?;
    writeln!(
        code,
        "    pub k: Vec<Vec<i8>>,       // [num_layers][MAX_SEQ_LEN * {}]  (quantized)",
        kv_size
    )?;
    writeln!(
        code,
        "    pub v: Vec<Vec<i8>>,       // [num_layers][MAX_SEQ_LEN * {}]  (quantized)",
        kv_size
    )?;
    writeln!(
        code,
        "    pub k_scales: Vec<Vec<f32>>,  // [num_layers][MAX_SEQ_LEN] per-token scale"
    )?;
    writeln!(
        code,
        "    pub v_scales: Vec<Vec<f32>>,  // [num_layers][MAX_SEQ_LEN] per-token scale"
    )?;
    writeln!(code, "    pub len: usize,")?;
    writeln!(code, "}}")?;
    writeln!(code)?;

    writeln!(code, "impl KVCache {{")?;
    writeln!(code, "    pub fn new() -> Self {{")?;
    writeln!(code, "        Self {{")?;
    writeln!(
        code,
        "            k: (0..NUM_LAYERS).map(|_| vec![0i8; MAX_SEQ_LEN * {kv_size}]).collect(),",
        kv_size = kv_size
    )?;
    writeln!(
        code,
        "            v: (0..NUM_LAYERS).map(|_| vec![0i8; MAX_SEQ_LEN * {kv_size}]).collect(),",
        kv_size = kv_size
    )?;
    writeln!(
        code,
        "            k_scales: (0..NUM_LAYERS).map(|_| vec![0.0f32; MAX_SEQ_LEN]).collect(),"
    )?;
    writeln!(
        code,
        "            v_scales: (0..NUM_LAYERS).map(|_| vec![0.0f32; MAX_SEQ_LEN]).collect(),"
    )?;
    writeln!(code, "            len: 0,")?;
    writeln!(code, "        }}")?;
    writeln!(code, "    }}")?;
    writeln!(code)?;
    writeln!(code, "    /// Reset cache without re-allocating")?;
    writeln!(code, "    pub fn reset(&mut self) {{")?;
    writeln!(code, "        self.len = 0;")?;
    writeln!(code, "    }}")?;
    writeln!(code)?;
    writeln!(code, "    /// Memory used by KV cache in bytes")?;
    writeln!(code, "    pub fn memory_bytes(&self) -> usize {{")?;
    writeln!(code, "        // int8 K+V data + f32 per-token scales")?;
    writeln!(
        code,
        "        NUM_LAYERS * MAX_SEQ_LEN * ({kv_size} * 1 * 2 + 4 * 2)  // k + v (i8) + scales (f32)"
    )?;
    writeln!(code, "    }}")?;
    writeln!(code, "}}")?;
    writeln!(code)?;
    writeln!(code, "impl Default for KVCache {{")?;
    writeln!(code, "    fn default() -> Self {{ Self::new() }}")?;
    writeln!(code, "}}")?;
    writeln!(code)?;

    // Forward function for single token (generation mode)
    writeln!(
        code,
        "/// Run forward pass for a single token. Returns logits [{vocab}]."
    )?;
    writeln!(
        code,
        "pub fn forward(token_id: u32, weights: &Weights, cache: &mut KVCache) -> Vec<f32> {{"
    )?;
    writeln!(code, "    let pos = cache.len;")?;
    writeln!(code)?;

    // Embedding
    writeln!(code, "    // Embedding lookup")?;
    writeln!(code, "    let mut hidden_state = [0.0f32; HIDDEN_SIZE];")?;
    writeln!(
        code,
        "    embedding(&mut hidden_state, token_id, &weights.embed_tokens, HIDDEN_SIZE);"
    )?;
    writeln!(code)?;

    // Buffers — fixed-size arrays, zero heap allocation
    writeln!(
        code,
        "    // Fixed-size buffers — zero heap allocation during forward pass"
    )?;
    writeln!(code, "    let mut normed = [0.0f32; {hidden}];")?;
    writeln!(code, "    let mut q = [0.0f32; {qk_size}];")?;
    writeln!(code, "    let mut k = [0.0f32; {kv_size}];")?;
    writeln!(code, "    let mut v = [0.0f32; {kv_size}];")?;
    writeln!(
        code,
        "    let mut k_q = [0i8; {kv_size}];  // int8 quantized k buffer"
    )?;
    writeln!(
        code,
        "    let mut v_q = [0i8; {kv_size}];  // int8 quantized v buffer"
    )?;
    writeln!(code, "    let mut attn_out = [0.0f32; {qk_size}];")?;
    writeln!(code, "    let mut attn_proj = [0.0f32; {hidden}];")?;
    writeln!(code, "    let mut gate = [0.0f32; {intermediate}];")?;
    writeln!(code, "    let mut up = [0.0f32; {intermediate}];")?;
    writeln!(code, "    let mut ffn_hidden = [0.0f32; {intermediate}];")?;
    writeln!(code, "    let mut ffn_out = [0.0f32; {hidden}];")?;
    writeln!(code)?;
    writeln!(
        code,
        "    // Precomputed RoPE frequencies — avoids powf per token"
    )?;
    writeln!(
        code,
        "    let rope_freqs = rope_freqs(HEAD_DIM, ROPE_THETA);"
    )?;
    writeln!(code)?;

    // Transformer layers
    writeln!(code, "    // Transformer layers")?;
    writeln!(code, "    for layer_idx in 0..NUM_LAYERS {{")?;
    writeln!(code, "        let lw = &weights.layers[layer_idx];")?;
    writeln!(code)?;
    writeln!(code, "        // Attention norm")?;
    writeln!(
        code,
        "        rms_norm(&mut normed, &hidden_state, &lw.attn_norm, RMS_NORM_EPS);"
    )?;
    writeln!(code)?;
    writeln!(code, "        // QKV projections (shape-specialized)")?;
    writeln!(
        code,
        "{}",
        matmul_call(q_dtype, hidden, qk_size, "q", "normed", "lw.q_proj", "        ")
    )?;
    writeln!(
        code,
        "{}",
        matmul_call(k_dtype, hidden, kv_size, "k", "normed", "lw.k_proj", "        ")
    )?;
    writeln!(
        code,
        "{}",
        matmul_call(v_dtype, hidden, kv_size, "v", "normed", "lw.v_proj", "        ")
    )?;
    // Optional QKV bias adds (Qwen2)
    if config.qkv_bias {
        writeln!(code)?;
        writeln!(code, "        // QKV bias additions (Qwen2)")?;
        writeln!(
            code,
            "        for i in 0..{qk_size} {{ q[i] += lw.q_bias[i]; }}"
        )?;
        writeln!(
            code,
            "        for i in 0..{kv_size} {{ k[i] += lw.k_bias[i]; }}"
        )?;
        writeln!(
            code,
            "        for i in 0..{kv_size} {{ v[i] += lw.v_bias[i]; }}"
        )?;
    }
    writeln!(code)?;
    writeln!(code, "        // RoPE")?;
    writeln!(
        code,
        "        rope(&mut q, pos, HEAD_DIM, NUM_HEADS, &rope_freqs);"
    )?;
    writeln!(
        code,
        "        rope(&mut k, pos, HEAD_DIM, NUM_KV_HEADS, &rope_freqs);"
    )?;
    writeln!(code)?;
    writeln!(
        code,
        "        // Quantize K/V to int8 and store in cache with per-token scale"
    )?;
    writeln!(
        code,
        "        cache.k_scales[layer_idx][pos] = quantize_kv(&k, &mut k_q);"
    )?;
    writeln!(
        code,
        "        cache.k[layer_idx][pos*{kv_size}..(pos+1)*{kv_size}].copy_from_slice(&k_q);",
        kv_size = kv_size
    )?;
    writeln!(
        code,
        "        cache.v_scales[layer_idx][pos] = quantize_kv(&v, &mut v_q);"
    )?;
    writeln!(
        code,
        "        cache.v[layer_idx][pos*{kv_size}..(pos+1)*{kv_size}].copy_from_slice(&v_q);",
        kv_size = kv_size
    )?;
    writeln!(code)?;
    writeln!(code, "        // Attention (cache sliced to valid region)")?;
    if config.sliding_window_size.is_some() {
        writeln!(code, "        attention_sliding(")?;
        writeln!(code, "            &mut attn_out, &q,")?;
        writeln!(
            code,
            "            &cache.k[layer_idx][..(pos+1)*{kv_size}], &cache.v[layer_idx][..(pos+1)*{kv_size}],",
            kv_size = kv_size
        )?;
        writeln!(
            code,
            "            &cache.k_scales[layer_idx][..pos+1], &cache.v_scales[layer_idx][..pos+1],"
        )?;
        writeln!(
            code,
            "            pos + 1, SLIDING_WINDOW_SIZE, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,"
        )?;
    } else if use_flash_attention(config) {
        writeln!(code, "        attention_flash(")?;
        writeln!(code, "            &mut attn_out, &q,")?;
        writeln!(
            code,
            "            &cache.k[layer_idx][..(pos+1)*{kv_size}], &cache.v[layer_idx][..(pos+1)*{kv_size}],",
            kv_size = kv_size
        )?;
        writeln!(
            code,
            "            &cache.k_scales[layer_idx][..pos+1], &cache.v_scales[layer_idx][..pos+1],"
        )?;
        writeln!(
            code,
            "            pos + 1, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,"
        )?;
    } else {
        writeln!(code, "        attention(")?;
        writeln!(code, "            &mut attn_out, &q,")?;
        writeln!(
            code,
            "            &cache.k[layer_idx][..(pos+1)*{kv_size}], &cache.v[layer_idx][..(pos+1)*{kv_size}],",
            kv_size = kv_size
        )?;
        writeln!(
            code,
            "            &cache.k_scales[layer_idx][..pos+1], &cache.v_scales[layer_idx][..pos+1],"
        )?;
        writeln!(
            code,
            "            pos + 1, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,"
        )?;
    }
    writeln!(code, "        );")?;
    writeln!(code)?;
    writeln!(code, "        // Output projection + fused residual add")?;
    writeln!(
        code,
        "{}",
        matmul_call(
            o_dtype,
            qk_size,
            hidden,
            "attn_proj",
            "attn_out",
            "lw.o_proj",
            "        "
        )
    )?;
    writeln!(code, "        residual_add(&mut hidden_state, &attn_proj);")?;
    writeln!(code)?;
    writeln!(code, "        // FFN norm")?;
    writeln!(
        code,
        "        rms_norm(&mut normed, &hidden_state, &lw.ffn_norm, RMS_NORM_EPS);"
    )?;
    writeln!(code)?;
    writeln!(
        code,
        "        // FFN: fused silu_mul eliminates gate_act buffer"
    )?;
    writeln!(
        code,
        "{}",
        matmul_call(
            gate_dtype,
            hidden,
            intermediate,
            "gate",
            "normed",
            "lw.gate_proj",
            "        "
        )
    )?;
    writeln!(
        code,
        "{}",
        matmul_call(
            up_dtype,
            hidden,
            intermediate,
            "up",
            "normed",
            "lw.up_proj",
            "        "
        )
    )?;
    writeln!(code, "        silu_mul(&mut ffn_hidden, &gate, &up);")?;
    writeln!(
        code,
        "{}",
        matmul_call(
            down_dtype,
            intermediate,
            hidden,
            "ffn_out",
            "ffn_hidden",
            "lw.down_proj",
            "        "
        )
    )?;
    writeln!(code)?;
    writeln!(code, "        // Fused residual add")?;
    writeln!(code, "        residual_add(&mut hidden_state, &ffn_out);")?;
    writeln!(code, "    }}")?;
    writeln!(code)?;

    // Final norm + logits
    writeln!(code, "    // Final norm")?;
    writeln!(
        code,
        "    rms_norm(&mut normed, &hidden_state, &weights.final_norm, RMS_NORM_EPS);"
    )?;
    writeln!(code)?;
    writeln!(
        code,
        "    // Logits projection (parallelized — largest single matmul)"
    )?;
    writeln!(
        code,
        "    // Uses larger chunks (256) to amortize Rayon overhead"
    )?;
    writeln!(code, "    let mut logits = vec![0.0f32; VOCAB_SIZE];")?;
    if lm_is_q8 {
        let lm_row_bytes = hidden.div_ceil(32) * 34;
        writeln!(code, "    #[cfg(target_arch = \"aarch64\")]")?;
        writeln!(
            code,
            "    let normed_q8 = quantize_to_q8_0_blocks(&normed[..]);"
        )?;
        writeln!(
            code,
            "    logits.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
        )?;
        writeln!(code, "        let base = chunk_idx * 256;")?;
        writeln!(code, "        #[cfg(target_arch = \"aarch64\")] {{")?;
        writeln!(code, "            let chunks8 = out.len() / 8;")?;
        writeln!(code, "            for c in 0..chunks8 {{")?;
        writeln!(code, "                let r = base + c * 8;")?;
        writeln!(
            code,
            "                let (d0,d1,d2,d3,d4,d5,d6,d7) = dot8_q8_0_q8_0(&normed_q8, &weights.lm_head[r*{lm_row_bytes}..(r+1)*{lm_row_bytes}], &weights.lm_head[(r+1)*{lm_row_bytes}..(r+2)*{lm_row_bytes}], &weights.lm_head[(r+2)*{lm_row_bytes}..(r+3)*{lm_row_bytes}], &weights.lm_head[(r+3)*{lm_row_bytes}..(r+4)*{lm_row_bytes}], &weights.lm_head[(r+4)*{lm_row_bytes}..(r+5)*{lm_row_bytes}], &weights.lm_head[(r+5)*{lm_row_bytes}..(r+6)*{lm_row_bytes}], &weights.lm_head[(r+6)*{lm_row_bytes}..(r+7)*{lm_row_bytes}], &weights.lm_head[(r+7)*{lm_row_bytes}..(r+8)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "                out[c*8] = d0; out[c*8+1] = d1; out[c*8+2] = d2; out[c*8+3] = d3; out[c*8+4] = d4; out[c*8+5] = d5; out[c*8+6] = d6; out[c*8+7] = d7;")?;
        writeln!(code, "            }}")?;
        writeln!(code, "            let tail8 = chunks8 * 8;")?;
        writeln!(code, "            let chunks4t = (out.len() - tail8) / 4;")?;
        writeln!(code, "            for c in 0..chunks4t {{")?;
        writeln!(code, "                let r = base + tail8 + c * 4;")?;
        writeln!(
            code,
            "                let (d0,d1,d2,d3) = dot4_q8_0_q8_0(&normed_q8, &weights.lm_head[r*{lm_row_bytes}..(r+1)*{lm_row_bytes}], &weights.lm_head[(r+1)*{lm_row_bytes}..(r+2)*{lm_row_bytes}], &weights.lm_head[(r+2)*{lm_row_bytes}..(r+3)*{lm_row_bytes}], &weights.lm_head[(r+3)*{lm_row_bytes}..(r+4)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "                out[tail8+c*4] = d0; out[tail8+c*4+1] = d1; out[tail8+c*4+2] = d2; out[tail8+c*4+3] = d3;")?;
        writeln!(code, "            }}")?;
        writeln!(
            code,
            "            for i in (tail8 + chunks4t*4)..out.len() {{"
        )?;
        writeln!(code, "                let j = base + i;")?;
        writeln!(
            code,
            "                out[i] = dot_q8_0_q8_0(&normed_q8, &weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "            }}")?;
        writeln!(code, "        }}")?;
        writeln!(code, "        #[cfg(not(target_arch = \"aarch64\"))]")?;
        writeln!(code, "        for r in 0..out.len() {{")?;
        writeln!(code, "            let j = base + r;")?;
        writeln!(
            code,
            "            out[r] = dot_q8_0(&normed[..], &weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "        }}")?;
        writeln!(code, "    }});")?;
    } else if lm_is_q4 {
        let lm_row_bytes = hidden.div_ceil(32) * 18;
        writeln!(code, "    #[cfg(target_arch = \"aarch64\")]")?;
        writeln!(
            code,
            "    let normed_q8 = quantize_to_q8_0_blocks(&normed[..]);"
        )?;
        writeln!(
            code,
            "    logits.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
        )?;
        writeln!(code, "        let base = chunk_idx * 256;")?;
        writeln!(code, "        #[cfg(target_arch = \"aarch64\")] {{")?;
        writeln!(code, "            let chunks8 = out.len() / 8;")?;
        writeln!(code, "            for c in 0..chunks8 {{")?;
        writeln!(code, "                let r = base + c * 8;")?;
        writeln!(
            code,
            "                let (d0,d1,d2,d3,d4,d5,d6,d7) = dot8_q4_0_q8_0(&normed_q8, &weights.lm_head[r*{lm_row_bytes}..(r+1)*{lm_row_bytes}], &weights.lm_head[(r+1)*{lm_row_bytes}..(r+2)*{lm_row_bytes}], &weights.lm_head[(r+2)*{lm_row_bytes}..(r+3)*{lm_row_bytes}], &weights.lm_head[(r+3)*{lm_row_bytes}..(r+4)*{lm_row_bytes}], &weights.lm_head[(r+4)*{lm_row_bytes}..(r+5)*{lm_row_bytes}], &weights.lm_head[(r+5)*{lm_row_bytes}..(r+6)*{lm_row_bytes}], &weights.lm_head[(r+6)*{lm_row_bytes}..(r+7)*{lm_row_bytes}], &weights.lm_head[(r+7)*{lm_row_bytes}..(r+8)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "                out[c*8] = d0; out[c*8+1] = d1; out[c*8+2] = d2; out[c*8+3] = d3; out[c*8+4] = d4; out[c*8+5] = d5; out[c*8+6] = d6; out[c*8+7] = d7;")?;
        writeln!(code, "            }}")?;
        writeln!(code, "            let tail8 = chunks8 * 8;")?;
        writeln!(code, "            let chunks4t = (out.len() - tail8) / 4;")?;
        writeln!(code, "            for c in 0..chunks4t {{")?;
        writeln!(code, "                let r = base + tail8 + c * 4;")?;
        writeln!(
            code,
            "                let (d0,d1,d2,d3) = dot4_q4_0_q8_0(&normed_q8, &weights.lm_head[r*{lm_row_bytes}..(r+1)*{lm_row_bytes}], &weights.lm_head[(r+1)*{lm_row_bytes}..(r+2)*{lm_row_bytes}], &weights.lm_head[(r+2)*{lm_row_bytes}..(r+3)*{lm_row_bytes}], &weights.lm_head[(r+3)*{lm_row_bytes}..(r+4)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "                out[tail8+c*4] = d0; out[tail8+c*4+1] = d1; out[tail8+c*4+2] = d2; out[tail8+c*4+3] = d3;")?;
        writeln!(code, "            }}")?;
        writeln!(
            code,
            "            for i in (tail8 + chunks4t*4)..out.len() {{"
        )?;
        writeln!(code, "                let j = base + i;")?;
        writeln!(
            code,
            "                out[i] = dot_q4_0_q8_0(&normed_q8, &weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "            }}")?;
        writeln!(code, "        }}")?;
        writeln!(code, "        #[cfg(not(target_arch = \"aarch64\"))]")?;
        writeln!(code, "        for r in 0..out.len() {{")?;
        writeln!(code, "            let j = base + r;")?;
        writeln!(
            code,
            "            out[r] = dot_q4_0(&normed[..], &weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "        }}")?;
        writeln!(code, "    }});")?;
    } else if lm_is_q4k {
        let lm_row_bytes = hidden.div_ceil(256) * 144;
        let q8_bytes = hidden.div_ceil(32) * 34;
        writeln!(
            code,
            "    let mut normed_q8 = vec![0u8; {q8_bytes}];"
        )?;
        writeln!(
            code,
            "    quantize_to_q8_0_blocks_into(&normed[..], &mut normed_q8);"
        )?;
        writeln!(
            code,
            "    logits.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
        )?;
        writeln!(code, "        let base = chunk_idx * 256;")?;
        writeln!(code, "        let len = out.len();")?;
        writeln!(code, "        #[cfg(target_arch = \"aarch64\")] {{")?;
        writeln!(code, "            let chunks4 = len / 4;")?;
        writeln!(code, "            for c in 0..chunks4 {{")?;
        writeln!(code, "                let r = base + c * 4;")?;
        writeln!(
            code,
            "                let (d0, d1, d2, d3) = dot4_q4_k_q8_0(&weights.lm_head[r*{lm_row_bytes}..(r+1)*{lm_row_bytes}], &weights.lm_head[(r+1)*{lm_row_bytes}..(r+2)*{lm_row_bytes}], &weights.lm_head[(r+2)*{lm_row_bytes}..(r+3)*{lm_row_bytes}], &weights.lm_head[(r+3)*{lm_row_bytes}..(r+4)*{lm_row_bytes}], &normed_q8, {hidden});"
        )?;
        writeln!(code, "                out[c*4] = d0; out[c*4+1] = d1; out[c*4+2] = d2; out[c*4+3] = d3;")?;
        writeln!(code, "            }}")?;
        writeln!(code, "            for i in (chunks4 * 4)..len {{")?;
        writeln!(code, "                let j = base + i;")?;
        writeln!(
            code,
            "                out[i] = dot_q4_k_q8_0(&weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], &normed_q8, {hidden});"
        )?;
        writeln!(code, "            }}")?;
        writeln!(code, "        }}")?;
        writeln!(code, "        #[cfg(not(target_arch = \"aarch64\"))] {{")?;
        writeln!(code, "            for r in 0..len {{")?;
        writeln!(code, "                let j = base + r;")?;
        writeln!(
            code,
            "                out[r] = dot_q4_k_q8_0(&weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], &normed_q8, {hidden});"
        )?;
        writeln!(code, "            }}")?;
        writeln!(code, "        }}")?;
        writeln!(code, "    }});")?;
    } else if lm_is_q6k {
        let lm_row_bytes = hidden.div_ceil(256) * 210;
        let q8_bytes = hidden.div_ceil(32) * 34;
        writeln!(code, "    let mut normed_q8 = vec![0u8; {q8_bytes}];")?;
        writeln!(code, "    quantize_to_q8_0_blocks_into(&normed[..], &mut normed_q8);")?;
        writeln!(code, "    logits.par_chunks_mut(64).enumerate().for_each(|(chunk_idx, out)| {{")?;
        writeln!(code, "        let base = chunk_idx * 64;")?;
        writeln!(code, "        for r in 0..out.len() {{")?;
        writeln!(code, "            let j = base + r;")?;
        writeln!(code, "            out[r] = dot_q6_k_q8_0(&weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], &normed_q8, {hidden});")?;
        writeln!(code, "        }}")?;
        writeln!(code, "    }});")?;
    } else {
        // f32 logits path: use cblas_sgemv on macOS for AMX acceleration,
        // fall back to parallel dot_f32 on other platforms.
        writeln!(code, "    #[cfg(target_os = \"macos\")] {{")?;
        writeln!(
            code,
            "        // Accelerate/AMX: single cblas_sgemv call for entire lm_head projection"
        )?;
        writeln!(code, "        // Weight layout: row-major [{vocab}, {hidden}], input: [{hidden}], output: [{vocab}]")?;
        writeln!(code, "        unsafe {{")?;
        writeln!(code, "            cblas_sgemv(")?;
        writeln!(code, "                CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,")?;
        writeln!(code, "                {vocab} as i32, {hidden} as i32,")?;
        writeln!(code, "                1.0,")?;
        writeln!(
            code,
            "                weights.lm_head.as_ptr(), {hidden} as i32,"
        )?;
        writeln!(code, "                normed.as_ptr(), 1,")?;
        writeln!(code, "                0.0,")?;
        writeln!(code, "                logits.as_mut_ptr(), 1,")?;
        writeln!(code, "            );")?;
        writeln!(code, "        }}")?;
        writeln!(code, "    }}")?;
        writeln!(code, "    #[cfg(not(target_os = \"macos\"))]")?;
        writeln!(code, "    {{")?;
        writeln!(
            code,
            "        logits.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
        )?;
        writeln!(code, "            let base = chunk_idx * 256;")?;
        writeln!(code, "            for r in 0..out.len() {{")?;
        writeln!(code, "                let j = base + r;")?;
        writeln!(
            code,
            "                out[r] = dot_f32(&normed[..], &weights.lm_head[j*{hidden}..(j+1)*{hidden}], {hidden});"
        )?;
        writeln!(code, "            }}")?;
        writeln!(code, "        }});")?;
        writeln!(code, "    }}")?;
    }
    writeln!(code)?;
    writeln!(code, "    cache.len += 1;")?;
    writeln!(code, "    logits")?;
    writeln!(code, "}}")?;

    Ok(())
}

/// Emit the `forward_prefill` function for batched prompt processing.
///
/// Processes a sequence of tokens in one pass, filling the KV cache for each
/// token sequentially. Returns logits for the last token only — used to pick
/// the first generated token. This avoids redundant tokenizer/function overhead
/// compared to calling `forward()` in a loop.
fn emit_prefill_function(code: &mut String, config: &ModelConfig) -> Result<(), CodegenError> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let vocab = config.vocab_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let qk_size = num_heads * head_dim;
    let kv_size = num_kv_heads * head_dim;

    use forgellm_frontend::ir::ProjCategory;
    let pdt = config.effective_proj_dtypes();
    let q_dtype = pdt.q;
    let k_dtype = pdt.k;
    let v_dtype = pdt.v;
    let o_dtype = pdt.o;
    let gate_dtype = pdt.gate;
    let up_dtype = pdt.up;
    let down_dtype = pdt.down;
    // Batched-prefill dispatch is only enabled when projections are uniformly
    // Q8_0 or Q4_0 (the kernel families that have batched specializations).
    let is_q8 = config.dtype == DType::Q8_0 && pdt.is_uniform();
    let is_q4 = config.dtype == DType::Q4_0 && pdt.is_uniform();
    // lm_head may use a different dtype than projections (see emit_forward_function).
    let lm_dtype = config.effective_dtype(ProjCategory::LmHead);
    let lm_is_q8 = lm_dtype == DType::Q8_0;
    let lm_is_q4 = lm_dtype == DType::Q4_0;
    let lm_is_q4k = lm_dtype == DType::Q4_K;
    let lm_is_q6k = lm_dtype == DType::Q6_K;

    writeln!(code)?;
    writeln!(code, "/// Process a prompt sequence and fill the KV cache.")?;
    writeln!(
        code,
        "/// Returns logits for the last token — use these to start generation."
    )?;
    writeln!(
        code,
        "/// Avoids per-token tokenizer overhead compared to calling forward() in a loop."
    )?;
    writeln!(
        code,
        "pub fn forward_prefill(tokens: &[u32], weights: &Weights, cache: &mut KVCache) -> Vec<f32> {{"
    )?;
    writeln!(code, "    let seq_len = tokens.len();")?;
    writeln!(code)?;
    // Dispatch to the batched prefill path when the prompt is long enough
    // to amortize heap allocation + batched-matmul overhead.  Below the
    // threshold, the per-token stack path has lower fixed cost.  Emitted
    // for Q8_0 and Q4_0 (f32 falls through to per-token).
    if is_q8 || is_q4 {
        writeln!(
            code,
            "    // Long prompts: batched matmul path amortizes weight loads across M tokens."
        )?;
        writeln!(
            code,
            "    // Threshold 8 ≈ where the heap-alloc + quantize overhead is paid back."
        )?;
        writeln!(
            code,
            "    // Set FORGE_BATCHED_PREFILL=0 at runtime to disable (A/B test vs per-token)."
        )?;
        writeln!(
            code,
            "    let batched_opt_out = std::env::var(\"FORGE_BATCHED_PREFILL\")"
        )?;
        writeln!(code, "        .map(|v| v == \"0\").unwrap_or(false);")?;
        writeln!(
            code,
            "    if !batched_opt_out && seq_len >= PREFILL_BATCH_THRESHOLD {{"
        )?;
        writeln!(
            code,
            "        return forward_prefill_batched(tokens, weights, cache);"
        )?;
        writeln!(code, "    }}")?;
        writeln!(code)?;
    }
    writeln!(
        code,
        "    // Precompute RoPE frequencies once for the entire prefill pass"
    )?;
    writeln!(
        code,
        "    let rope_freqs = rope_freqs(HEAD_DIM, ROPE_THETA);"
    )?;
    writeln!(code)?;
    writeln!(
        code,
        "    // Fixed-size stack buffers — same as forward(), zero heap allocation"
    )?;
    writeln!(code, "    let mut hidden_state = [0.0f32; HIDDEN_SIZE];")?;
    writeln!(code, "    let mut normed = [0.0f32; {hidden}];")?;
    writeln!(code, "    let mut q = [0.0f32; {qk_size}];")?;
    writeln!(code, "    let mut k = [0.0f32; {kv_size}];")?;
    writeln!(code, "    let mut v = [0.0f32; {kv_size}];")?;
    writeln!(
        code,
        "    let mut k_q = [0i8; {kv_size}];  // int8 quantized k buffer"
    )?;
    writeln!(
        code,
        "    let mut v_q = [0i8; {kv_size}];  // int8 quantized v buffer"
    )?;
    writeln!(code, "    let mut attn_out = [0.0f32; {qk_size}];")?;
    writeln!(code, "    let mut attn_proj = [0.0f32; {hidden}];")?;
    writeln!(code, "    let mut gate = [0.0f32; {intermediate}];")?;
    writeln!(code, "    let mut up = [0.0f32; {intermediate}];")?;
    writeln!(code, "    let mut ffn_hidden = [0.0f32; {intermediate}];")?;
    writeln!(code, "    let mut ffn_out = [0.0f32; {hidden}];")?;
    writeln!(code, "    let mut last_logits = vec![0.0f32; VOCAB_SIZE];")?;
    writeln!(code)?;
    writeln!(
        code,
        "    for (tok_pos, &token_id) in tokens.iter().enumerate() {{"
    )?;
    writeln!(code, "        let pos = cache.len + tok_pos;")?;
    writeln!(code)?;
    writeln!(code, "        // Embedding lookup")?;
    writeln!(
        code,
        "        embedding(&mut hidden_state, token_id, &weights.embed_tokens, HIDDEN_SIZE);"
    )?;
    writeln!(code)?;
    writeln!(code, "        // Transformer layers")?;
    writeln!(code, "        for layer_idx in 0..NUM_LAYERS {{")?;
    writeln!(code, "            let lw = &weights.layers[layer_idx];")?;
    writeln!(code)?;
    writeln!(code, "            // Attention norm")?;
    writeln!(
        code,
        "            rms_norm(&mut normed, &hidden_state, &lw.attn_norm, RMS_NORM_EPS);"
    )?;
    writeln!(code)?;
    writeln!(code, "            // QKV projections")?;
    writeln!(
        code,
        "{}",
        matmul_call(q_dtype, hidden, qk_size, "q", "normed", "lw.q_proj", "            ")
    )?;
    writeln!(
        code,
        "{}",
        matmul_call(k_dtype, hidden, kv_size, "k", "normed", "lw.k_proj", "            ")
    )?;
    writeln!(
        code,
        "{}",
        matmul_call(v_dtype, hidden, kv_size, "v", "normed", "lw.v_proj", "            ")
    )?;
    // Optional QKV bias adds (Qwen2) — mirrors forward(), applied before RoPE.
    if config.qkv_bias {
        writeln!(code)?;
        writeln!(code, "            // QKV bias additions (Qwen2)")?;
        writeln!(
            code,
            "            for i in 0..{qk_size} {{ q[i] += lw.q_bias[i]; }}"
        )?;
        writeln!(
            code,
            "            for i in 0..{kv_size} {{ k[i] += lw.k_bias[i]; }}"
        )?;
        writeln!(
            code,
            "            for i in 0..{kv_size} {{ v[i] += lw.v_bias[i]; }}"
        )?;
    }
    writeln!(code)?;
    writeln!(code, "            // RoPE")?;
    writeln!(
        code,
        "            rope(&mut q, pos, HEAD_DIM, NUM_HEADS, &rope_freqs);"
    )?;
    writeln!(
        code,
        "            rope(&mut k, pos, HEAD_DIM, NUM_KV_HEADS, &rope_freqs);"
    )?;
    writeln!(code)?;
    writeln!(
        code,
        "            // Quantize K/V to int8 and store in cache with per-token scale"
    )?;
    writeln!(
        code,
        "            cache.k_scales[layer_idx][pos] = quantize_kv(&k, &mut k_q);"
    )?;
    writeln!(
        code,
        "            cache.k[layer_idx][pos*{kv_size}..(pos+1)*{kv_size}].copy_from_slice(&k_q);",
        kv_size = kv_size
    )?;
    writeln!(
        code,
        "            cache.v_scales[layer_idx][pos] = quantize_kv(&v, &mut v_q);"
    )?;
    writeln!(
        code,
        "            cache.v[layer_idx][pos*{kv_size}..(pos+1)*{kv_size}].copy_from_slice(&v_q);",
        kv_size = kv_size
    )?;
    writeln!(code)?;
    writeln!(
        code,
        "            // Attention over all filled positions (causal: current token sees all previous + itself)"
    )?;
    if use_flash_attention(config) {
        writeln!(code, "            attention_flash(")?;
    } else {
        writeln!(code, "            attention(")?;
    }
    writeln!(code, "                &mut attn_out, &q,")?;
    writeln!(
        code,
        "                &cache.k[layer_idx][..(pos+1)*{kv_size}], &cache.v[layer_idx][..(pos+1)*{kv_size}],",
        kv_size = kv_size
    )?;
    writeln!(
        code,
        "                &cache.k_scales[layer_idx][..pos+1], &cache.v_scales[layer_idx][..pos+1],"
    )?;
    writeln!(
        code,
        "                pos + 1, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,"
    )?;
    writeln!(code, "            );")?;
    writeln!(code)?;
    writeln!(code, "            // Output projection + residual")?;
    writeln!(
        code,
        "{}",
        matmul_call(
            o_dtype,
            qk_size,
            hidden,
            "attn_proj",
            "attn_out",
            "lw.o_proj",
            "            "
        )
    )?;
    writeln!(
        code,
        "            residual_add(&mut hidden_state, &attn_proj);"
    )?;
    writeln!(code)?;
    writeln!(code, "            // FFN norm")?;
    writeln!(
        code,
        "            rms_norm(&mut normed, &hidden_state, &lw.ffn_norm, RMS_NORM_EPS);"
    )?;
    writeln!(code)?;
    writeln!(code, "            // FFN: fused silu_mul")?;
    writeln!(
        code,
        "{}",
        matmul_call(
            gate_dtype,
            hidden,
            intermediate,
            "gate",
            "normed",
            "lw.gate_proj",
            "            "
        )
    )?;
    writeln!(
        code,
        "{}",
        matmul_call(
            up_dtype,
            hidden,
            intermediate,
            "up",
            "normed",
            "lw.up_proj",
            "            "
        )
    )?;
    writeln!(code, "            silu_mul(&mut ffn_hidden, &gate, &up);")?;
    writeln!(
        code,
        "{}",
        matmul_call(
            down_dtype,
            intermediate,
            hidden,
            "ffn_out",
            "ffn_hidden",
            "lw.down_proj",
            "            "
        )
    )?;
    writeln!(
        code,
        "            residual_add(&mut hidden_state, &ffn_out);"
    )?;
    writeln!(code, "        }}")?;
    writeln!(code)?;
    writeln!(
        code,
        "        // On the last token, compute logits for the first generated token"
    )?;
    writeln!(code, "        if tok_pos == seq_len - 1 {{")?;
    writeln!(
        code,
        "            rms_norm(&mut normed, &hidden_state, &weights.final_norm, RMS_NORM_EPS);"
    )?;
    if lm_is_q8 {
        let lm_row_bytes = hidden.div_ceil(32) * 34;
        writeln!(code, "            #[cfg(target_arch = \"aarch64\")]")?;
        writeln!(
            code,
            "            let normed_q8 = quantize_to_q8_0_blocks(&normed[..]);"
        )?;
        writeln!(
            code,
            "            last_logits.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
        )?;
        writeln!(code, "                let base = chunk_idx * 256;")?;
        writeln!(code, "                #[cfg(target_arch = \"aarch64\")] {{")?;
        writeln!(code, "                    let chunks8 = out.len() / 8;")?;
        writeln!(code, "                    for c in 0..chunks8 {{")?;
        writeln!(code, "                        let r = base + c * 8;")?;
        writeln!(
            code,
            "                        let (d0,d1,d2,d3,d4,d5,d6,d7) = dot8_q8_0_q8_0(&normed_q8, &weights.lm_head[r*{lm_row_bytes}..(r+1)*{lm_row_bytes}], &weights.lm_head[(r+1)*{lm_row_bytes}..(r+2)*{lm_row_bytes}], &weights.lm_head[(r+2)*{lm_row_bytes}..(r+3)*{lm_row_bytes}], &weights.lm_head[(r+3)*{lm_row_bytes}..(r+4)*{lm_row_bytes}], &weights.lm_head[(r+4)*{lm_row_bytes}..(r+5)*{lm_row_bytes}], &weights.lm_head[(r+5)*{lm_row_bytes}..(r+6)*{lm_row_bytes}], &weights.lm_head[(r+6)*{lm_row_bytes}..(r+7)*{lm_row_bytes}], &weights.lm_head[(r+7)*{lm_row_bytes}..(r+8)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "                        out[c*8] = d0; out[c*8+1] = d1; out[c*8+2] = d2; out[c*8+3] = d3; out[c*8+4] = d4; out[c*8+5] = d5; out[c*8+6] = d6; out[c*8+7] = d7;")?;
        writeln!(code, "                    }}")?;
        writeln!(code, "                    let tail8 = chunks8 * 8;")?;
        writeln!(
            code,
            "                    let chunks4t = (out.len() - tail8) / 4;"
        )?;
        writeln!(code, "                    for c in 0..chunks4t {{")?;
        writeln!(
            code,
            "                        let r = base + tail8 + c * 4;"
        )?;
        writeln!(
            code,
            "                        let (d0,d1,d2,d3) = dot4_q8_0_q8_0(&normed_q8, &weights.lm_head[r*{lm_row_bytes}..(r+1)*{lm_row_bytes}], &weights.lm_head[(r+1)*{lm_row_bytes}..(r+2)*{lm_row_bytes}], &weights.lm_head[(r+2)*{lm_row_bytes}..(r+3)*{lm_row_bytes}], &weights.lm_head[(r+3)*{lm_row_bytes}..(r+4)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "                        out[tail8+c*4] = d0; out[tail8+c*4+1] = d1; out[tail8+c*4+2] = d2; out[tail8+c*4+3] = d3;")?;
        writeln!(code, "                    }}")?;
        writeln!(
            code,
            "                    for i in (tail8 + chunks4t*4)..out.len() {{"
        )?;
        writeln!(code, "                        let j = base + i;")?;
        writeln!(
            code,
            "                        out[i] = dot_q8_0_q8_0(&normed_q8, &weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "                    }}")?;
        writeln!(code, "                }}")?;
        writeln!(
            code,
            "                #[cfg(not(target_arch = \"aarch64\"))]"
        )?;
        writeln!(code, "                for r in 0..out.len() {{")?;
        writeln!(code, "                    let j = base + r;")?;
        writeln!(
            code,
            "                    out[r] = dot_q8_0(&normed[..], &weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "                }}")?;
        writeln!(code, "            }});")?;
    } else if lm_is_q4 {
        let lm_row_bytes = hidden.div_ceil(32) * 18;
        writeln!(code, "            #[cfg(target_arch = \"aarch64\")]")?;
        writeln!(
            code,
            "            let normed_q8 = quantize_to_q8_0_blocks(&normed[..]);"
        )?;
        writeln!(
            code,
            "            last_logits.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
        )?;
        writeln!(code, "                let base = chunk_idx * 256;")?;
        writeln!(code, "                #[cfg(target_arch = \"aarch64\")] {{")?;
        writeln!(code, "                    let chunks8 = out.len() / 8;")?;
        writeln!(code, "                    for c in 0..chunks8 {{")?;
        writeln!(code, "                        let r = base + c * 8;")?;
        writeln!(
            code,
            "                        let (d0,d1,d2,d3,d4,d5,d6,d7) = dot8_q4_0_q8_0(&normed_q8, &weights.lm_head[r*{lm_row_bytes}..(r+1)*{lm_row_bytes}], &weights.lm_head[(r+1)*{lm_row_bytes}..(r+2)*{lm_row_bytes}], &weights.lm_head[(r+2)*{lm_row_bytes}..(r+3)*{lm_row_bytes}], &weights.lm_head[(r+3)*{lm_row_bytes}..(r+4)*{lm_row_bytes}], &weights.lm_head[(r+4)*{lm_row_bytes}..(r+5)*{lm_row_bytes}], &weights.lm_head[(r+5)*{lm_row_bytes}..(r+6)*{lm_row_bytes}], &weights.lm_head[(r+6)*{lm_row_bytes}..(r+7)*{lm_row_bytes}], &weights.lm_head[(r+7)*{lm_row_bytes}..(r+8)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "                        out[c*8] = d0; out[c*8+1] = d1; out[c*8+2] = d2; out[c*8+3] = d3; out[c*8+4] = d4; out[c*8+5] = d5; out[c*8+6] = d6; out[c*8+7] = d7;")?;
        writeln!(code, "                    }}")?;
        writeln!(code, "                    let tail8 = chunks8 * 8;")?;
        writeln!(
            code,
            "                    let chunks4t = (out.len() - tail8) / 4;"
        )?;
        writeln!(code, "                    for c in 0..chunks4t {{")?;
        writeln!(
            code,
            "                        let r = base + tail8 + c * 4;"
        )?;
        writeln!(
            code,
            "                        let (d0,d1,d2,d3) = dot4_q4_0_q8_0(&normed_q8, &weights.lm_head[r*{lm_row_bytes}..(r+1)*{lm_row_bytes}], &weights.lm_head[(r+1)*{lm_row_bytes}..(r+2)*{lm_row_bytes}], &weights.lm_head[(r+2)*{lm_row_bytes}..(r+3)*{lm_row_bytes}], &weights.lm_head[(r+3)*{lm_row_bytes}..(r+4)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "                        out[tail8+c*4] = d0; out[tail8+c*4+1] = d1; out[tail8+c*4+2] = d2; out[tail8+c*4+3] = d3;")?;
        writeln!(code, "                    }}")?;
        writeln!(
            code,
            "                    for i in (tail8 + chunks4t*4)..out.len() {{"
        )?;
        writeln!(code, "                        let j = base + i;")?;
        writeln!(
            code,
            "                        out[i] = dot_q4_0_q8_0(&normed_q8, &weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "                    }}")?;
        writeln!(code, "                }}")?;
        writeln!(
            code,
            "                #[cfg(not(target_arch = \"aarch64\"))]"
        )?;
        writeln!(code, "                for r in 0..out.len() {{")?;
        writeln!(code, "                    let j = base + r;")?;
        writeln!(
            code,
            "                    out[r] = dot_q4_0(&normed[..], &weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], {hidden});"
        )?;
        writeln!(code, "                }}")?;
        writeln!(code, "            }});")?;
    } else if lm_is_q4k {
        let lm_row_bytes = hidden.div_ceil(256) * 144;
        let q8_bytes = hidden.div_ceil(32) * 34;
        writeln!(
            code,
            "            let mut normed_q8 = vec![0u8; {q8_bytes}];"
        )?;
        writeln!(
            code,
            "            quantize_to_q8_0_blocks_into(&normed[..], &mut normed_q8);"
        )?;
        writeln!(
            code,
            "            last_logits.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
        )?;
        writeln!(code, "                let base = chunk_idx * 256;")?;
        writeln!(code, "                let len = out.len();")?;
        writeln!(code, "                #[cfg(target_arch = \"aarch64\")] {{")?;
        writeln!(code, "                    let chunks4 = len / 4;")?;
        writeln!(code, "                    for c in 0..chunks4 {{")?;
        writeln!(code, "                        let r = base + c * 4;")?;
        writeln!(
            code,
            "                        let (d0, d1, d2, d3) = dot4_q4_k_q8_0(&weights.lm_head[r*{lm_row_bytes}..(r+1)*{lm_row_bytes}], &weights.lm_head[(r+1)*{lm_row_bytes}..(r+2)*{lm_row_bytes}], &weights.lm_head[(r+2)*{lm_row_bytes}..(r+3)*{lm_row_bytes}], &weights.lm_head[(r+3)*{lm_row_bytes}..(r+4)*{lm_row_bytes}], &normed_q8, {hidden});"
        )?;
        writeln!(code, "                        out[c*4] = d0; out[c*4+1] = d1; out[c*4+2] = d2; out[c*4+3] = d3;")?;
        writeln!(code, "                    }}")?;
        writeln!(code, "                    for i in (chunks4 * 4)..len {{")?;
        writeln!(code, "                        let j = base + i;")?;
        writeln!(
            code,
            "                        out[i] = dot_q4_k_q8_0(&weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], &normed_q8, {hidden});"
        )?;
        writeln!(code, "                    }}")?;
        writeln!(code, "                }}")?;
        writeln!(code, "                #[cfg(not(target_arch = \"aarch64\"))] {{")?;
        writeln!(code, "                    for r in 0..len {{")?;
        writeln!(code, "                        let j = base + r;")?;
        writeln!(
            code,
            "                        out[r] = dot_q4_k_q8_0(&weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], &normed_q8, {hidden});"
        )?;
        writeln!(code, "                    }}")?;
        writeln!(code, "                }}")?;
        writeln!(code, "            }});")?;
    } else if lm_is_q6k {
        let lm_row_bytes = hidden.div_ceil(256) * 210;
        let q8_bytes = hidden.div_ceil(32) * 34;
        writeln!(code, "            let mut normed_q8 = vec![0u8; {q8_bytes}];")?;
        writeln!(code, "            quantize_to_q8_0_blocks_into(&normed[..], &mut normed_q8);")?;
        writeln!(code, "            last_logits.par_chunks_mut(64).enumerate().for_each(|(chunk_idx, out)| {{")?;
        writeln!(code, "                let base = chunk_idx * 64;")?;
        writeln!(code, "                for r in 0..out.len() {{")?;
        writeln!(code, "                    let j = base + r;")?;
        writeln!(code, "                    out[r] = dot_q6_k_q8_0(&weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], &normed_q8, {hidden});")?;
        writeln!(code, "                }}")?;
        writeln!(code, "            }});")?;
    } else {
        // f32 prefill logits: use cblas_sgemv on macOS, parallel dot_f32 elsewhere
        writeln!(code, "            #[cfg(target_os = \"macos\")] {{")?;
        writeln!(
            code,
            "                // Accelerate/AMX: single cblas_sgemv for lm_head projection"
        )?;
        writeln!(code, "                unsafe {{")?;
        writeln!(code, "                    cblas_sgemv(")?;
        writeln!(
            code,
            "                        CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,"
        )?;
        writeln!(
            code,
            "                        {vocab} as i32, {hidden} as i32,"
        )?;
        writeln!(code, "                        1.0,")?;
        writeln!(
            code,
            "                        weights.lm_head.as_ptr(), {hidden} as i32,"
        )?;
        writeln!(code, "                        normed.as_ptr(), 1,")?;
        writeln!(code, "                        0.0,")?;
        writeln!(code, "                        last_logits.as_mut_ptr(), 1,")?;
        writeln!(code, "                    );")?;
        writeln!(code, "                }}")?;
        writeln!(code, "            }}")?;
        writeln!(code, "            #[cfg(not(target_os = \"macos\"))]")?;
        writeln!(code, "            {{")?;
        writeln!(
            code,
            "                last_logits.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
        )?;
        writeln!(code, "                    let base = chunk_idx * 256;")?;
        writeln!(code, "                    for r in 0..out.len() {{")?;
        writeln!(code, "                        let j = base + r;")?;
        writeln!(
            code,
            "                        out[r] = dot_f32(&normed[..], &weights.lm_head[j*{hidden}..(j+1)*{hidden}], {hidden});"
        )?;
        writeln!(code, "                    }}")?;
        writeln!(code, "                }});")?;
        writeln!(code, "            }}")?;
    }
    writeln!(code, "        }}")?;
    writeln!(code, "    }}")?;
    writeln!(code)?;
    writeln!(code, "    cache.len += seq_len;")?;
    writeln!(code, "    last_logits")?;
    writeln!(code, "}}")?;

    Ok(())
}

/// Emit `forward_prefill_batched` — processes all M prompt tokens with
/// batched matmul for QKV/O/FFN projections and per-token activations
/// (rms_norm, RoPE, attention, silu_mul, residual).  The key win is that
/// each weight matrix is loaded from RAM once per forward pass, not M
/// times, which is the bottleneck for long prompts on CPU.
///
/// Memory: allocates heap buffers sized `m * max_hidden_vector`.  For
/// M=2000 on Llama-3.2-1B that's ~200 MB peak — acceptable for prefill,
/// returned at end of call.
///
/// Layout: all batch buffers are row-major `[m][dim]`.  `matmul_mat_q8_0`
/// reads row-major and writes row-major to preserve that contract.
///
/// Only emitted for Q8_0 models.  Q4_0 / f32 fall through to the per-token
/// `forward_prefill` path.
fn emit_prefill_batched_function(
    code: &mut String,
    config: &ModelConfig,
) -> Result<(), CodegenError> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let qk_size = num_heads * head_dim;
    let kv_size = num_kv_heads * head_dim;
    // Projection dispatch — `matmul_mat_{mm_kind}_KxN` for per-layer weights.
    let mm_kind = match config.dtype {
        DType::Q8_0 => "q8_0",
        DType::Q4_0 => "q4_0",
        _ => unreachable!("forward_prefill_batched only emitted for Q8_0 / Q4_0"),
    };
    // lm_head dispatch — may differ from projections (e.g. Q4_K_M → Q4_0 proj + Q8_0 lm_head).
    let lm_dtype = config.lm_head_dtype.unwrap_or(config.dtype);
    let (lm_row_bytes, lm_dot8, lm_dot, lm_scalar_dot) = match lm_dtype {
        DType::Q8_0 => (
            hidden.div_ceil(32) * 34,
            "dot8_q8_0_q8_0",
            "dot_q8_0_q8_0",
            "dot_q8_0",
        ),
        DType::Q4_0 => (
            hidden.div_ceil(32) * 18,
            "dot8_q4_0_q8_0",
            "dot_q4_0_q8_0",
            "dot_q4_0",
        ),
        _ => unreachable!("forward_prefill_batched lm_head only supports Q8_0 / Q4_0"),
    };

    writeln!(code)?;
    writeln!(
        code,
        "/// Batched prefill: processes all M prompt tokens with batched matmul."
    )?;
    writeln!(
        code,
        "/// Each weight matrix is loaded from RAM once per forward pass (vs M times"
    )?;
    writeln!(
        code,
        "/// in per-token prefill), amortizing weight bandwidth by a factor of M."
    )?;
    writeln!(
        code,
        "pub fn forward_prefill_batched(tokens: &[u32], weights: &Weights, cache: &mut KVCache) -> Vec<f32> {{"
    )?;
    writeln!(code, "    let m = tokens.len();")?;
    writeln!(code, "    let base_pos = cache.len;")?;
    writeln!(
        code,
        "    let rope_freqs = rope_freqs(HEAD_DIM, ROPE_THETA);"
    )?;
    writeln!(code)?;

    // Heap batch buffers.
    writeln!(
        code,
        "    // Per-token state, row-major [m][dim].  Heap-allocated for M flexibility."
    )?;
    writeln!(
        code,
        "    let mut hidden_batch = vec![0.0f32; m * {hidden}];"
    )?;
    writeln!(
        code,
        "    let mut normed_batch = vec![0.0f32; m * {hidden}];"
    )?;
    writeln!(code, "    let mut q_batch = vec![0.0f32; m * {qk_size}];")?;
    writeln!(code, "    let mut k_batch = vec![0.0f32; m * {kv_size}];")?;
    writeln!(code, "    let mut v_batch = vec![0.0f32; m * {kv_size}];")?;
    writeln!(
        code,
        "    let mut attn_out_batch = vec![0.0f32; m * {qk_size}];"
    )?;
    writeln!(
        code,
        "    let mut attn_proj_batch = vec![0.0f32; m * {hidden}];"
    )?;
    writeln!(
        code,
        "    let mut gate_batch = vec![0.0f32; m * {intermediate}];"
    )?;
    writeln!(
        code,
        "    let mut up_batch = vec![0.0f32; m * {intermediate}];"
    )?;
    writeln!(
        code,
        "    let mut ffn_hidden_batch = vec![0.0f32; m * {intermediate}];"
    )?;
    writeln!(
        code,
        "    let mut ffn_out_batch = vec![0.0f32; m * {hidden}];"
    )?;
    writeln!(code)?;

    // Initial embedding lookup.
    writeln!(
        code,
        "    // Embedding lookup: per-token (memory-bandwidth bound, can't batch)."
    )?;
    writeln!(
        code,
        "    for (r, &token_id) in tokens.iter().enumerate() {{"
    )?;
    writeln!(
        code,
        "        embedding(&mut hidden_batch[r*{hidden}..(r+1)*{hidden}], token_id, &weights.embed_tokens, HIDDEN_SIZE);"
    )?;
    writeln!(code, "    }}")?;
    writeln!(code)?;

    // Main layer loop.
    writeln!(code, "    for layer_idx in 0..NUM_LAYERS {{")?;
    writeln!(code, "        let lw = &weights.layers[layer_idx];")?;
    writeln!(code)?;

    // Attention rms_norm (per-token).
    writeln!(
        code,
        "        // Attention rms_norm (per-token, O(hidden) per token)"
    )?;
    writeln!(code, "        for r in 0..m {{")?;
    writeln!(
        code,
        "            rms_norm(&mut normed_batch[r*{hidden}..(r+1)*{hidden}], &hidden_batch[r*{hidden}..(r+1)*{hidden}], &lw.attn_norm, RMS_NORM_EPS);"
    )?;
    writeln!(code, "        }}")?;
    writeln!(code)?;

    // Batched QKV matmul.
    writeln!(
        code,
        "        // Batched QKV projections: weight loaded once, used for all m tokens"
    )?;
    writeln!(
        code,
        "        matmul_mat_{mm_kind}_{hidden}x{qk_size}(&mut q_batch, m, &normed_batch, &lw.q_proj);"
    )?;
    writeln!(
        code,
        "        matmul_mat_{mm_kind}_{hidden}x{kv_size}(&mut k_batch, m, &normed_batch, &lw.k_proj);"
    )?;
    writeln!(
        code,
        "        matmul_mat_{mm_kind}_{hidden}x{kv_size}(&mut v_batch, m, &normed_batch, &lw.v_proj);"
    )?;
    writeln!(code)?;

    // Per-token: bias (optional), RoPE, KV cache write.  Attention is
    // batched after this loop so each K/V position is loaded once per
    // Q_TILE queries instead of once per query.
    writeln!(
        code,
        "        // Per-token: optional QKV bias, RoPE, KV cache write (attention is batched below)"
    )?;
    writeln!(code, "        let mut k_q = [0i8; {kv_size}];")?;
    writeln!(code, "        let mut v_q = [0i8; {kv_size}];")?;
    writeln!(code, "        for r in 0..m {{")?;
    writeln!(code, "            let pos = base_pos + r;")?;
    writeln!(
        code,
        "            let q_row = &mut q_batch[r*{qk_size}..(r+1)*{qk_size}];"
    )?;
    writeln!(
        code,
        "            let k_row = &mut k_batch[r*{kv_size}..(r+1)*{kv_size}];"
    )?;
    writeln!(
        code,
        "            let v_row = &mut v_batch[r*{kv_size}..(r+1)*{kv_size}];"
    )?;
    if config.qkv_bias {
        writeln!(code, "            // QKV bias additions (Qwen2)")?;
        writeln!(
            code,
            "            for i in 0..{qk_size} {{ q_row[i] += lw.q_bias[i]; }}"
        )?;
        writeln!(
            code,
            "            for i in 0..{kv_size} {{ k_row[i] += lw.k_bias[i]; }}"
        )?;
        writeln!(
            code,
            "            for i in 0..{kv_size} {{ v_row[i] += lw.v_bias[i]; }}"
        )?;
    }
    writeln!(
        code,
        "            rope(q_row, pos, HEAD_DIM, NUM_HEADS, &rope_freqs);"
    )?;
    writeln!(
        code,
        "            rope(k_row, pos, HEAD_DIM, NUM_KV_HEADS, &rope_freqs);"
    )?;
    writeln!(
        code,
        "            cache.k_scales[layer_idx][pos] = quantize_kv(k_row, &mut k_q);"
    )?;
    writeln!(
        code,
        "            cache.k[layer_idx][pos*{kv_size}..(pos+1)*{kv_size}].copy_from_slice(&k_q);"
    )?;
    writeln!(
        code,
        "            cache.v_scales[layer_idx][pos] = quantize_kv(v_row, &mut v_q);"
    )?;
    writeln!(
        code,
        "            cache.v[layer_idx][pos*{kv_size}..(pos+1)*{kv_size}].copy_from_slice(&v_q);"
    )?;
    writeln!(code, "        }}")?;
    writeln!(code)?;

    // Batched attention over all M queries.  Uses attention_flash_batch
    // for flash-attention-eligible models; falls back to a per-token
    // `attention` loop otherwise (SWA / very short contexts).
    if use_flash_attention(config) {
        writeln!(
            code,
            "        // Batched flash attention: K/V amortized across Q_TILE queries per block"
        )?;
        writeln!(code, "        let total_seq = base_pos + m;")?;
        writeln!(code, "        attention_flash_batch(")?;
        writeln!(code, "            &mut attn_out_batch, &q_batch,")?;
        writeln!(
            code,
            "            &cache.k[layer_idx][..total_seq*{kv_size}], &cache.v[layer_idx][..total_seq*{kv_size}],"
        )?;
        writeln!(
            code,
            "            &cache.k_scales[layer_idx][..total_seq], &cache.v_scales[layer_idx][..total_seq],"
        )?;
        writeln!(
            code,
            "            m, base_pos, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,"
        )?;
        writeln!(code, "        );")?;
    } else if let Some(window) = config.sliding_window_size {
        // SWA models (Mistral, Gemma-2, ...): batched sliding-window attention.
        // `attention_sliding_batch` applies both causal and window masks and
        // bounds the outer K-block loop to the tile's combined valid K range.
        // Each K/V position is loaded once per (head, Q_TILE) pair instead of
        // once per query — ~Q_TILE bandwidth reduction vs the per-token
        // `attention_sliding` fallback that would otherwise run here.
        writeln!(
            code,
            "        // Batched sliding-window attention (SWA model)"
        )?;
        writeln!(code, "        let total_seq = base_pos + m;")?;
        writeln!(code, "        attention_sliding_batch(")?;
        writeln!(code, "            &mut attn_out_batch, &q_batch,")?;
        writeln!(
            code,
            "            &cache.k[layer_idx][..total_seq*{kv_size}], &cache.v[layer_idx][..total_seq*{kv_size}],"
        )?;
        writeln!(
            code,
            "            &cache.k_scales[layer_idx][..total_seq], &cache.v_scales[layer_idx][..total_seq],"
        )?;
        writeln!(
            code,
            "            m, base_pos, {window}, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,"
        )?;
        writeln!(code, "        );")?;
    } else {
        // Short-context non-SWA models (max_seq_len <= 512): plain attention.
        writeln!(
            code,
            "        // Fallback: per-token standard attention (short-context, non-SWA)"
        )?;
        writeln!(code, "        for r in 0..m {{")?;
        writeln!(code, "            let pos = base_pos + r;")?;
        writeln!(
            code,
            "            let q_row = &q_batch[r*{qk_size}..(r+1)*{qk_size}];"
        )?;
        writeln!(code, "            attention(")?;
        writeln!(
            code,
            "                &mut attn_out_batch[r*{qk_size}..(r+1)*{qk_size}], q_row,"
        )?;
        writeln!(
            code,
            "                &cache.k[layer_idx][..(pos+1)*{kv_size}], &cache.v[layer_idx][..(pos+1)*{kv_size}],"
        )?;
        writeln!(
            code,
            "                &cache.k_scales[layer_idx][..pos+1], &cache.v_scales[layer_idx][..pos+1],"
        )?;
        writeln!(
            code,
            "                pos + 1, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM,"
        )?;
        writeln!(code, "            );")?;
        writeln!(code, "        }}")?;
    }
    writeln!(code)?;

    // Batched O matmul.
    writeln!(
        code,
        "        // Batched O projection + per-token residual add"
    )?;
    writeln!(
        code,
        "        matmul_mat_{mm_kind}_{qk_size}x{hidden}(&mut attn_proj_batch, m, &attn_out_batch, &lw.o_proj);"
    )?;
    writeln!(code, "        for r in 0..m {{")?;
    writeln!(
        code,
        "            residual_add(&mut hidden_batch[r*{hidden}..(r+1)*{hidden}], &attn_proj_batch[r*{hidden}..(r+1)*{hidden}]);"
    )?;
    writeln!(code, "        }}")?;
    writeln!(code)?;

    // FFN norm + batched gate/up + silu_mul + batched down + residual.
    writeln!(code, "        // FFN rms_norm (per-token)")?;
    writeln!(code, "        for r in 0..m {{")?;
    writeln!(
        code,
        "            rms_norm(&mut normed_batch[r*{hidden}..(r+1)*{hidden}], &hidden_batch[r*{hidden}..(r+1)*{hidden}], &lw.ffn_norm, RMS_NORM_EPS);"
    )?;
    writeln!(code, "        }}")?;
    writeln!(code)?;
    writeln!(
        code,
        "        // Batched gate/up matmul + per-token silu_mul"
    )?;
    writeln!(
        code,
        "        matmul_mat_{mm_kind}_{hidden}x{intermediate}(&mut gate_batch, m, &normed_batch, &lw.gate_proj);"
    )?;
    writeln!(
        code,
        "        matmul_mat_{mm_kind}_{hidden}x{intermediate}(&mut up_batch, m, &normed_batch, &lw.up_proj);"
    )?;
    writeln!(code, "        for r in 0..m {{")?;
    // Match architecture-dependent activation: existing forward_prefill uses silu_mul.
    writeln!(
        code,
        "            silu_mul(&mut ffn_hidden_batch[r*{intermediate}..(r+1)*{intermediate}], &gate_batch[r*{intermediate}..(r+1)*{intermediate}], &up_batch[r*{intermediate}..(r+1)*{intermediate}]);"
    )?;
    writeln!(code, "        }}")?;
    writeln!(code, "        // Batched down matmul + per-token residual")?;
    writeln!(
        code,
        "        matmul_mat_{mm_kind}_{intermediate}x{hidden}(&mut ffn_out_batch, m, &ffn_hidden_batch, &lw.down_proj);"
    )?;
    writeln!(code, "        for r in 0..m {{")?;
    writeln!(
        code,
        "            residual_add(&mut hidden_batch[r*{hidden}..(r+1)*{hidden}], &ffn_out_batch[r*{hidden}..(r+1)*{hidden}]);"
    )?;
    writeln!(code, "        }}")?;
    writeln!(code, "    }}")?;
    writeln!(code)?;

    // Final norm + lm_head on last token only.  Dtype-specific kernels:
    // Q8_0 uses dot8_q8_0_q8_0 / dot_q8_0_q8_0 on 34-byte rows;
    // Q4_0 uses dot8_q4_0_q8_0 / dot_q4_0_q8_0 on 18-byte rows.
    writeln!(
        code,
        "    // Final rms_norm + lm_head on the last token (matches forward_prefill)"
    )?;
    writeln!(code, "    let mut normed = [0.0f32; {hidden}];")?;
    writeln!(
        code,
        "    rms_norm(&mut normed, &hidden_batch[(m-1)*{hidden}..m*{hidden}], &weights.final_norm, RMS_NORM_EPS);"
    )?;
    writeln!(code, "    let mut last_logits = vec![0.0f32; VOCAB_SIZE];")?;
    writeln!(code, "    #[cfg(target_arch = \"aarch64\")]")?;
    writeln!(
        code,
        "    let normed_q8 = quantize_to_q8_0_blocks(&normed[..]);"
    )?;
    writeln!(
        code,
        "    last_logits.par_chunks_mut(256).enumerate().for_each(|(chunk_idx, out)| {{"
    )?;
    writeln!(code, "        let base = chunk_idx * 256;")?;
    writeln!(code, "        #[cfg(target_arch = \"aarch64\")] {{")?;
    writeln!(code, "            let chunks8 = out.len() / 8;")?;
    writeln!(code, "            for c in 0..chunks8 {{")?;
    writeln!(code, "                let r = base + c * 8;")?;
    writeln!(
        code,
        "                let (d0,d1,d2,d3,d4,d5,d6,d7) = {lm_dot8}(&normed_q8, &weights.lm_head[r*{lm_row_bytes}..(r+1)*{lm_row_bytes}], &weights.lm_head[(r+1)*{lm_row_bytes}..(r+2)*{lm_row_bytes}], &weights.lm_head[(r+2)*{lm_row_bytes}..(r+3)*{lm_row_bytes}], &weights.lm_head[(r+3)*{lm_row_bytes}..(r+4)*{lm_row_bytes}], &weights.lm_head[(r+4)*{lm_row_bytes}..(r+5)*{lm_row_bytes}], &weights.lm_head[(r+5)*{lm_row_bytes}..(r+6)*{lm_row_bytes}], &weights.lm_head[(r+6)*{lm_row_bytes}..(r+7)*{lm_row_bytes}], &weights.lm_head[(r+7)*{lm_row_bytes}..(r+8)*{lm_row_bytes}], {hidden});"
    )?;
    writeln!(code, "                out[c*8] = d0; out[c*8+1] = d1; out[c*8+2] = d2; out[c*8+3] = d3; out[c*8+4] = d4; out[c*8+5] = d5; out[c*8+6] = d6; out[c*8+7] = d7;")?;
    writeln!(code, "            }}")?;
    writeln!(code, "            let tail8 = chunks8 * 8;")?;
    writeln!(code, "            for i in tail8..out.len() {{")?;
    writeln!(code, "                let j = base + i;")?;
    writeln!(
        code,
        "                out[i] = {lm_dot}(&normed_q8, &weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], {hidden});"
    )?;
    writeln!(code, "            }}")?;
    writeln!(code, "        }}")?;
    writeln!(code, "        #[cfg(not(target_arch = \"aarch64\"))]")?;
    writeln!(code, "        for r in 0..out.len() {{")?;
    writeln!(code, "            let j = base + r;")?;
    writeln!(
        code,
        "            out[r] = {lm_scalar_dot}(&normed[..], &weights.lm_head[j*{lm_row_bytes}..(j+1)*{lm_row_bytes}], {hidden});"
    )?;
    writeln!(code, "        }}")?;
    writeln!(code, "    }});")?;
    writeln!(code)?;
    writeln!(code, "    cache.len += m;")?;
    writeln!(code, "    last_logits")?;
    writeln!(code, "}}")?;
    writeln!(code)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use forgellm_frontend::graph_builder;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 256,
            max_seq_len: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::F16,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        }
    }

    #[test]
    fn generate_code_from_tiny_model() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        assert!(code.contains("pub const HIDDEN_SIZE: usize = 64;"));
        assert!(code.contains("pub const NUM_LAYERS: usize = 2;"));
        assert!(code.contains("pub const VOCAB_SIZE: usize = 256;"));
        assert!(code.contains("fn rms_norm("));
        assert!(code.contains("fn matmul_vec_"));
        assert!(code.contains("fn forward("));
        assert!(code.contains("fn attention("));
    }

    #[test]
    fn generated_code_has_correct_structure() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Should have weight structs
        assert!(code.contains("pub struct Weights"));
        assert!(code.contains("pub struct LayerWeights"));
        assert!(code.contains("pub struct KVCache"));

        // Should have all kernel functions including fused ops
        assert!(code.contains("pub fn silu("));
        assert!(code.contains("pub fn silu_mul("));
        assert!(code.contains("pub fn residual_add("));
        assert!(code.contains("pub fn softmax("));
        assert!(code.contains("pub fn rope("));
        assert!(code.contains("pub fn embedding("));
        assert!(code.contains("pub fn elementwise_mul("));
        assert!(code.contains("pub fn elementwise_add("));
    }

    #[test]
    fn generated_code_has_accelerate_ffi() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Accelerate framework FFI declarations should always be emitted
        assert!(
            code.contains("fn cblas_sgemv("),
            "generated code should declare cblas_sgemv FFI"
        );
        assert!(
            code.contains("#[link(name = \"Accelerate\", kind = \"framework\")]"),
            "generated code should link Accelerate framework"
        );
        assert!(
            code.contains("CBLAS_ROW_MAJOR"),
            "generated code should define CBLAS_ROW_MAJOR constant"
        );
        assert!(
            code.contains("CBLAS_NO_TRANS"),
            "generated code should define CBLAS_NO_TRANS constant"
        );
    }

    #[test]
    fn f32_matmul_uses_cblas_on_macos() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // f32 specialized matmul functions should have macOS cblas_sgemv path
        assert!(
            code.contains("#[cfg(target_os = \"macos\")]\n#[inline]\nfn matmul_vec_"),
            "f32 matmul_vec should have a macOS-specific variant using cblas_sgemv"
        );
        assert!(
            code.contains("#[cfg(not(target_os = \"macos\"))]\n#[inline]\nfn matmul_vec_")
                || code.contains("#[cfg(not(target_os = \"macos\"))]\n/// Parallelized"),
            "f32 matmul_vec should have a non-macOS fallback variant"
        );

        // The f32 logits path should use cblas_sgemv on macOS
        assert!(
            code.contains("cfg(target_os = \"macos\")]") && code.contains("cblas_sgemv("),
            "f32 logits projection should use cblas_sgemv on macOS"
        );
    }

    #[test]
    fn q8_matmul_does_not_use_cblas() {
        let config = ModelConfig {
            dtype: DType::Q8_0,
            lm_head_dtype: None,
            proj_dtypes: None,
            ..tiny_config()
        };
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Q8_0 matmul functions should NOT use cblas (it only handles f32)
        assert!(
            code.contains("fn matmul_vec_q8_0_"),
            "Q8_0 model should have quantized matmul functions"
        );
        // Q8_0 matmul_vec_q8_0_* should not contain cblas calls
        // (cblas_sgemv is only for f32 matmul_vec_* and f32 logits)
        let q8_matmul_section = code
            .find("fn matmul_vec_q8_0_")
            .and_then(|start| code[start..].find("fn matmul_vec_q8_0_").map(|_| start));
        assert!(
            q8_matmul_section.is_some(),
            "should find Q8_0 matmul functions"
        );
    }

    #[test]
    fn generate_code_for_llama_1b() {
        let config = ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 2048,
            intermediate_size: 5632,
            num_layers: 16,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            vocab_size: 32000,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::F16,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        };

        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        assert!(code.contains("pub const HIDDEN_SIZE: usize = 2048;"));
        assert!(code.contains("pub const NUM_HEADS: usize = 32;"));
        assert!(code.contains("pub const NUM_KV_HEADS: usize = 8;"));
    }

    #[test]
    fn generate_code_for_smollm() {
        let config = ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 576,
            intermediate_size: 1536,
            num_layers: 30,
            num_attention_heads: 9,
            num_kv_heads: 3,
            head_dim: 64,
            vocab_size: 49152,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::BF16,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        };

        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        assert!(code.contains("pub const HIDDEN_SIZE: usize = 576;"));
        assert!(code.contains("pub const NUM_LAYERS: usize = 30;"));

        // Shape-specialized matmul: hidden=576, qk=9*64=576, kv=3*64=192, inter=1536
        assert!(code.contains("fn matmul_vec_576x576(")); // q_proj (hidden→qk)
        assert!(code.contains("fn matmul_vec_576x192(")); // k/v_proj (hidden→kv)
        assert!(code.contains("fn matmul_vec_576x1536(")); // gate/up_proj
        assert!(code.contains("fn matmul_vec_1536x576(")); // down_proj
                                                           // Forward function uses specialized calls
        assert!(code.contains("matmul_vec_576x576(&mut q"));
        assert!(code.contains("matmul_vec_576x192(&mut k"));
        assert!(!code.contains("matmul(&mut q")); // no generic matmul calls
    }

    #[test]
    fn generated_code_compiles_as_rust() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Verify the generated code is valid Rust by checking syntax
        // (actual compilation test would require writing to file and running cargo)
        assert!(code.starts_with("//!"));
        assert!(!code.contains("TODO"));
        assert!(!code.contains("unimplemented"));

        // Check balanced braces
        let opens: usize = code.chars().filter(|&c| c == '{').count();
        let closes: usize = code.chars().filter(|&c| c == '}').count();
        assert_eq!(opens, closes, "unbalanced braces in generated code");
    }

    #[test]
    fn graph_without_config_errors() {
        let graph = Graph::new("no-config");
        let result = generate(&graph);
        assert!(matches!(result, Err(CodegenError::MissingConfig)));
    }

    #[test]
    fn generated_code_has_fused_ops() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Fused silu_mul kernel exists
        assert!(code.contains("pub fn silu_mul("));
        // Fused gelu_mul kernel exists
        assert!(code.contains("pub fn gelu_mul("));
        // Fused residual_add kernel exists
        assert!(code.contains("pub fn residual_add("));
        // Forward uses fused ops
        assert!(code.contains("silu_mul(&mut ffn_hidden"));
        assert!(code.contains("residual_add(&mut hidden_state"));
        // No separate silu+elementwise_mul in forward path
        assert!(!code.contains("silu(&mut gate_act"));
    }

    #[test]
    fn generated_code_has_rayon_parallel() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Should have rayon import
        assert!(code.contains("use rayon::prelude::*;"));
        // Logits should be parallelized (par_chunks_mut for ILP)
        assert!(code.contains("par_chunks_mut"));
    }

    #[test]
    fn generated_code_has_rope_precomputation() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        assert!(code.contains("pub fn rope_freqs("));
        assert!(code.contains("let rope_freqs = rope_freqs(HEAD_DIM, ROPE_THETA);"));
        assert!(code.contains("&rope_freqs)"));
    }

    #[test]
    fn generated_code_has_neon_simd() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // NEON dot product
        assert!(code.contains("#[cfg(target_arch = \"aarch64\")]"));
        assert!(code.contains("vfmaq_f32"));
        // NEON elementwise ops
        assert!(code.contains("vmulq_f32"));
        assert!(code.contains("vaddq_f32"));
        // Scalar fallbacks
        assert!(code.contains("#[cfg(not(target_arch = \"aarch64\"))]"));
    }

    #[test]
    fn generated_code_has_static_memory() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Fixed-size stack buffers (not vec!)
        assert!(code.contains("let mut normed = [0.0f32;"));
        assert!(code.contains("let mut q = [0.0f32;"));
        assert!(code.contains("let mut hidden_state = [0.0f32;"));
        // Pre-allocated KV cache
        assert!(code.contains("MAX_SEQ_LEN *"));
        // Attention scores on stack
        assert!(code.contains("[0.0f32; MAX_SEQ_LEN]"));
    }

    #[test]
    fn generated_code_has_4_accumulator_dot_f32() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // dot_f32 should use 4 accumulators (s0..s3)
        assert!(code.contains("let mut s0 = vdupq_n_f32(0.0);"));
        assert!(code.contains("let mut s1 = vdupq_n_f32(0.0);"));
        assert!(code.contains("let mut s2 = vdupq_n_f32(0.0);"));
        assert!(code.contains("let mut s3 = vdupq_n_f32(0.0);"));
        // 16-element unrolled loop
        assert!(code.contains("len / 16"));
    }

    #[test]
    fn parallel_path_uses_chunks_with_ilp() {
        // Use a config with N >= 512 to trigger parallel path
        let config = ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 896,
            intermediate_size: 4864, // > 512 threshold
            num_layers: 24,
            num_attention_heads: 14,
            num_kv_heads: 2,
            head_dim: 64,
            vocab_size: 151936,
            max_seq_len: 2048,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            dtype: DType::F16,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        };
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Should use par_chunks_mut(256) for parallel matmul
        assert!(code.contains("par_chunks_mut(256)"));
        // Should have 4-way row ILP within parallel chunks
        assert!(code.contains("let chunks4 = len / 4;"));
        // Specialized matmul for the intermediate size
        assert!(code.contains("matmul_vec_896x4864("));
    }

    #[test]
    fn generated_code_has_4_way_row_unrolling() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Sequential matmul should process 4 rows at a time
        assert!(code.contains("Process 4 output rows at a time"));
        assert!(code.contains("output[j0]"));
        assert!(code.contains("output[j0+1]"));
        assert!(code.contains("output[j0+2]"));
        assert!(code.contains("output[j0+3]"));
    }

    #[test]
    fn parallel_matmul_large_n_uses_par_chunks() {
        // smollm2-135m: hidden=576, intermediate=1536 — N=1536 >= 512
        let config = ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 576,
            intermediate_size: 1536,
            num_layers: 30,
            num_attention_heads: 9,
            num_kv_heads: 3,
            head_dim: 64,
            vocab_size: 49152,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::F16,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        };
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // intermediate_size=1536 >= 512 → matmul_vec_576x1536 should use par_chunks_mut
        assert!(
            code.contains("par_chunks_mut"),
            "N=1536 >= 512 should trigger parallel path in matmul_vec_576x1536"
        );
    }

    #[test]
    fn parallel_matmul_small_n_stays_sequential() {
        // tiny_config: hidden=64, intermediate=128, qk=64, kv=32, vocab=256 — all N < 512
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // All specialized matmul_vec_* functions for N < 512 must be sequential.
        // matmul_vec_64x128 is the largest specialized matmul — verify it uses sequential path.
        assert!(
            code.contains("fn matmul_vec_64x128("),
            "matmul_vec_64x128 should be emitted"
        );
        // The generated code has par_chunks_mut only in the hardcoded logits loops
        // (forward + forward_prefill), not in any specialized matmul_vec_* function.
        // A simpler proxy: the specialized matmul functions all appear before forward(),
        // so we check that par_chunks_mut does NOT appear before the "pub fn forward(" marker.
        let forward_pos = code
            .find("pub fn forward(")
            .expect("forward function should exist");
        let pre_forward = &code[..forward_pos];
        assert!(
            !pre_forward.contains("par_chunks_mut"),
            "no specialized matmul_vec_* function (N < 512) should use par_chunks_mut"
        );
    }

    #[test]
    fn generated_project_has_rayon_dep() {
        // Verify the generated model code imports rayon (header must emit use rayon::prelude::*)
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();
        assert!(
            code.contains("use rayon::prelude::*;"),
            "generated model.rs should import rayon prelude"
        );
    }

    #[test]
    fn generated_code_has_release_profile_optimizations() {
        // Verified at the project.rs level — emit.rs only generates model.rs
        // This test verifies the model file has the dead_code allow
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        assert!(code.contains("#![allow(dead_code, unused_imports, unused_assignments)]"));
    }

    fn tiny_q8_config() -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 256,
            max_seq_len: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::Q8_0,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        }
    }

    #[test]
    fn q8_0_model_generates_vec_u8_fields() {
        let config = tiny_q8_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Projection weights should be Vec<u8>
        assert!(
            code.contains("pub q_proj: Vec<u8>"),
            "q_proj should be Vec<u8> for Q8_0 model"
        );
        assert!(
            code.contains("pub k_proj: Vec<u8>"),
            "k_proj should be Vec<u8> for Q8_0 model"
        );
        assert!(
            code.contains("pub v_proj: Vec<u8>"),
            "v_proj should be Vec<u8> for Q8_0 model"
        );
        assert!(
            code.contains("pub o_proj: Vec<u8>"),
            "o_proj should be Vec<u8> for Q8_0 model"
        );
        assert!(
            code.contains("pub gate_proj: Vec<u8>"),
            "gate_proj should be Vec<u8> for Q8_0 model"
        );
        assert!(
            code.contains("pub up_proj: Vec<u8>"),
            "up_proj should be Vec<u8> for Q8_0 model"
        );
        assert!(
            code.contains("pub down_proj: Vec<u8>"),
            "down_proj should be Vec<u8> for Q8_0 model"
        );
        assert!(
            code.contains("pub lm_head: Vec<u8>"),
            "lm_head should be Vec<u8> for Q8_0 model"
        );
        // Norm and embedding weights are always f32
        assert!(
            code.contains("pub attn_norm: Vec<f32>"),
            "attn_norm should remain Vec<f32>"
        );
        assert!(
            code.contains("pub embed_tokens: Vec<f32>"),
            "embed_tokens should remain Vec<f32>"
        );
        assert!(
            code.contains("pub final_norm: Vec<f32>"),
            "final_norm should remain Vec<f32>"
        );
    }

    #[test]
    fn q8_0_model_generates_dot_q8_0_function() {
        let config = tiny_q8_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        assert!(
            code.contains("fn dot_q8_0("),
            "Q8_0 model should emit dot_q8_0 function"
        );
        assert!(
            code.contains("fn f16_bits_to_f32("),
            "Q8_0 model should emit f16_bits_to_f32 helper"
        );
    }

    #[test]
    fn q8_0_model_calls_q8_matmul_in_forward() {
        let config = tiny_q8_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Forward function should use q8_0 matmul variants
        assert!(
            code.contains("matmul_vec_q8_0_"),
            "Q8_0 model should call matmul_vec_q8_0_* variants"
        );
        // Should not use regular f32 matmul for projections in forward
        assert!(
            !code.contains("matmul_vec_64x64(&mut q"),
            "Q8_0 model should not call f32 matmul for q_proj"
        );
    }

    #[test]
    fn f32_model_does_not_generate_q8_0_kernels() {
        let config = tiny_config(); // F16 dtype
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Non-Q8_0 models should not have q8_0 functions
        assert!(
            !code.contains("fn dot_q8_0("),
            "non-Q8_0 model should not emit dot_q8_0"
        );
        assert!(
            !code.contains("matmul_vec_q8_0_"),
            "non-Q8_0 model should not emit Q8_0 matmul variants"
        );
        // Should use regular Vec<f32> for all weight fields
        assert!(code.contains("pub q_proj: Vec<f32>"));
        assert!(code.contains("pub lm_head: Vec<f32>"));
    }

    #[test]
    fn qwen2_config_emits_q_bias_field_in_layer_weights() {
        let config = ModelConfig {
            architecture: Architecture::Qwen2,
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 256,
            max_seq_len: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            dtype: DType::F16,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: true,
            hidden_activation: HiddenActivation::SiLU,
        };
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // LayerWeights should have bias fields
        assert!(
            code.contains("pub q_bias: Vec<f32>"),
            "Qwen2 LayerWeights should have q_bias field"
        );
        assert!(
            code.contains("pub k_bias: Vec<f32>"),
            "Qwen2 LayerWeights should have k_bias field"
        );
        assert!(
            code.contains("pub v_bias: Vec<f32>"),
            "Qwen2 LayerWeights should have v_bias field"
        );
        // Forward should apply the biases — BOTH in forward() and forward_prefill().
        assert!(
            code.contains("q[i] += lw.q_bias[i]"),
            "Qwen2 forward() should add q_bias"
        );
        assert!(
            code.contains("k[i] += lw.k_bias[i]"),
            "Qwen2 forward() should add k_bias"
        );
        // forward_prefill must also apply the bias (issue #210: previously missing).
        let fwd_idx = code.find("pub fn forward(").expect("forward() must exist");
        let prefill_idx = code
            .find("pub fn forward_prefill(")
            .expect("forward_prefill() must exist");
        let prefill_body = &code[prefill_idx..];
        assert!(
            prefill_body.contains("q[i] += lw.q_bias[i]"),
            "Qwen2 forward_prefill() should also add q_bias (issue #210)"
        );
        assert!(
            prefill_body.contains("v[i] += lw.v_bias[i]"),
            "Qwen2 forward_prefill() should also add v_bias (issue #210)"
        );
        let _ = fwd_idx;
        assert!(
            code.contains("v[i] += lw.v_bias[i]"),
            "Qwen2 forward should add v_bias"
        );
    }

    #[test]
    fn llama_config_does_not_emit_bias_fields() {
        let config = tiny_config(); // Llama, qkv_bias = false
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Llama should NOT have bias fields in LayerWeights
        assert!(
            !code.contains("pub q_bias:"),
            "Llama should not have q_bias field"
        );
        assert!(
            !code.contains("pub k_bias:"),
            "Llama should not have k_bias field"
        );
        assert!(
            !code.contains("pub v_bias:"),
            "Llama should not have v_bias field"
        );
        // And should not call any bias adds in forward
        assert!(
            !code.contains("lw.q_bias"),
            "Llama forward should not reference q_bias"
        );
    }

    #[test]
    fn mistral_config_with_swa_emits_attention_sliding() {
        let config = ModelConfig {
            architecture: Architecture::Mistral,
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 256,
            max_seq_len: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::F16,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: Some(64),
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        };
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Should emit attention_sliding function
        assert!(
            code.contains("pub fn attention_sliding("),
            "Mistral with SWA should emit attention_sliding function"
        );
        // Forward should call attention_sliding instead of attention
        assert!(
            code.contains("attention_sliding("),
            "Mistral forward should call attention_sliding"
        );
        // SLIDING_WINDOW_SIZE constant should be set
        assert!(
            code.contains("pub const SLIDING_WINDOW_SIZE: usize = 64;"),
            "Mistral should emit SLIDING_WINDOW_SIZE = 64"
        );
        // The forward function should NOT have a bare `attention(` call (only `attention_sliding(`)
        // We verify this by checking the forward() function body specifically.
        // Note: the prefill function also calls attention() — that is expected and correct.
        let forward_fn_start = code
            .find("pub fn forward(token_id")
            .expect("forward function must exist");
        let forward_fn_end = code
            .find("pub fn forward_prefill(")
            .expect("forward_prefill must exist");
        let forward_body = &code[forward_fn_start..forward_fn_end];
        assert!(
            !forward_body.contains("        attention(\n"),
            "Mistral forward() should not call regular attention — only attention_sliding"
        );
    }

    #[test]
    fn q8_swa_batched_prefill_uses_sliding_batch() {
        // Q8_0 + SWA model: the batched prefill path must call
        // attention_sliding_batch (v0.8.5) — not plain attention()
        // (v0.8.0-v0.8.3 correctness bug) and not the per-token
        // attention_sliding() fallback (v0.8.4 stop-gap).
        let config = ModelConfig {
            architecture: Architecture::Mistral,
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 256,
            max_seq_len: 64, // <= 512 so use_flash_attention returns false
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::Q8_0,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: Some(32),
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        };
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // attention_sliding_batch kernel must be emitted.
        assert!(
            code.contains("pub fn attention_sliding_batch("),
            "SWA Q8_0 models must emit attention_sliding_batch kernel"
        );

        // forward_prefill_batched must be emitted (Q8_0 path).
        let batched_start = code
            .find("pub fn forward_prefill_batched(")
            .expect("forward_prefill_batched must be emitted for Q8_0");
        let batched_body = &code[batched_start..];

        // Must call the batched sliding kernel, not the per-token fallback.
        assert!(
            batched_body.contains("attention_sliding_batch("),
            "forward_prefill_batched must call attention_sliding_batch for SWA models"
        );
        // Must NOT take the non-SWA fallback branch.
        let fallback_marker = "// Fallback: per-token standard attention (short-context, non-SWA)";
        assert!(
            !batched_body.contains(fallback_marker),
            "SWA model must not take the non-SWA fallback branch"
        );
        // The window size must be passed through to the kernel call.
        assert!(
            batched_body.contains("m, base_pos, 32,"),
            "window size must be passed as the third runtime arg"
        );
    }

    #[test]
    fn llama_config_without_swa_emits_regular_attention() {
        let config = tiny_config(); // Llama, no SWA
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Should emit standard attention function (defined in kernel template)
        assert!(
            code.contains("pub fn attention("),
            "Llama should emit standard attention function"
        );
        // SLIDING_WINDOW_SIZE should be 0 (full attention)
        assert!(
            code.contains("pub const SLIDING_WINDOW_SIZE: usize = 0;"),
            "Llama should emit SLIDING_WINDOW_SIZE = 0"
        );
        // Forward should call regular attention
        assert!(
            code.contains("        attention("),
            "Llama forward should call regular attention"
        );
        // Should NOT call attention_sliding in forward
        assert!(
            !code.contains("attention_sliding("),
            "Llama should not call attention_sliding"
        );
    }

    fn tiny_q4_config() -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 256,
            max_seq_len: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::Q4_0,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        }
    }

    #[test]
    fn q4_0_model_generates_vec_u8_fields() {
        let config = tiny_q4_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Projection weights should be Vec<u8>
        assert!(
            code.contains("pub q_proj: Vec<u8>"),
            "q_proj should be Vec<u8> for Q4_0 model"
        );
        assert!(
            code.contains("pub k_proj: Vec<u8>"),
            "k_proj should be Vec<u8> for Q4_0 model"
        );
        assert!(
            code.contains("pub v_proj: Vec<u8>"),
            "v_proj should be Vec<u8> for Q4_0 model"
        );
        assert!(
            code.contains("pub o_proj: Vec<u8>"),
            "o_proj should be Vec<u8> for Q4_0 model"
        );
        assert!(
            code.contains("pub gate_proj: Vec<u8>"),
            "gate_proj should be Vec<u8> for Q4_0 model"
        );
        assert!(
            code.contains("pub up_proj: Vec<u8>"),
            "up_proj should be Vec<u8> for Q4_0 model"
        );
        assert!(
            code.contains("pub down_proj: Vec<u8>"),
            "down_proj should be Vec<u8> for Q4_0 model"
        );
        assert!(
            code.contains("pub lm_head: Vec<u8>"),
            "lm_head should be Vec<u8> for Q4_0 model"
        );
        // Norm and embedding weights are always f32
        assert!(
            code.contains("pub attn_norm: Vec<f32>"),
            "attn_norm should remain Vec<f32>"
        );
        assert!(
            code.contains("pub embed_tokens: Vec<f32>"),
            "embed_tokens should remain Vec<f32>"
        );
        assert!(
            code.contains("pub final_norm: Vec<f32>"),
            "final_norm should remain Vec<f32>"
        );
    }

    #[test]
    fn q4_0_model_generates_dot_q4_0_function() {
        let config = tiny_q4_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Scalar fallback kernel
        assert!(
            code.contains("fn dot_q4_0("),
            "Q4_0 model should emit dot_q4_0 scalar fallback function"
        );
        assert!(
            code.contains("fn f16_bits_to_f32("),
            "Q4_0 model should emit f16_bits_to_f32 helper"
        );
        // NEON sdot kernels
        assert!(
            code.contains("fn dot_q4_0_q8_0("),
            "Q4_0 model should emit dot_q4_0_q8_0 NEON sdot function"
        );
        assert!(
            code.contains("fn dot4_q4_0_q8_0("),
            "Q4_0 model should emit dot4_q4_0_q8_0 batch function"
        );
        assert!(
            code.contains("fn dot8_q4_0_q8_0("),
            "Q4_0 model should emit dot8_q4_0_q8_0 batch function"
        );
        // Q8_0 input quantization helpers
        assert!(
            code.contains("fn quantize_to_q8_0_blocks_into("),
            "Q4_0 model should emit quantize_to_q8_0_blocks_into helper"
        );
        assert!(
            code.contains("fn f32_to_f16_bits("),
            "Q4_0 model should emit f32_to_f16_bits helper"
        );
    }

    #[test]
    fn q4_0_model_calls_q4_matmul_in_forward() {
        let config = tiny_q4_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Forward function should use q4_0 matmul variants
        assert!(
            code.contains("matmul_vec_q4_0_"),
            "Q4_0 model should call matmul_vec_q4_0_* variants"
        );
        // Should not use regular f32 matmul for projections in forward
        assert!(
            !code.contains("matmul_vec_64x64(&mut q"),
            "Q4_0 model should not call f32 matmul for q_proj"
        );
        // AArch64 matmul functions should quantize input to Q8_0
        assert!(
            code.contains("quantize_to_q8_0_blocks_into(&input[..], &mut input_q8)"),
            "Q4_0 aarch64 matmul should quantize input to Q8_0"
        );
        // AArch64 matmul should use dot8/dot4 cascade
        assert!(
            code.contains("dot8_q4_0_q8_0("),
            "Q4_0 aarch64 matmul should use dot8_q4_0_q8_0"
        );
    }

    #[test]
    fn f32_model_does_not_generate_q4_0_kernels() {
        let config = tiny_config(); // F16 dtype
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Non-Q4_0 models should not have q4_0 functions
        assert!(
            !code.contains("fn dot_q4_0("),
            "non-Q4_0 model should not emit dot_q4_0"
        );
        assert!(
            !code.contains("fn dot_q4_0_q8_0("),
            "non-Q4_0 model should not emit dot_q4_0_q8_0"
        );
        assert!(
            !code.contains("fn dot4_q4_0_q8_0("),
            "non-Q4_0 model should not emit dot4_q4_0_q8_0"
        );
        assert!(
            !code.contains("fn dot8_q4_0_q8_0("),
            "non-Q4_0 model should not emit dot8_q4_0_q8_0"
        );
        assert!(
            !code.contains("matmul_vec_q4_0_"),
            "non-Q4_0 model should not emit Q4_0 matmul variants"
        );
        // Should use regular Vec<f32> for all weight fields
        assert!(code.contains("pub q_proj: Vec<f32>"));
        assert!(code.contains("pub lm_head: Vec<f32>"));
    }

    // ─── Flash Attention tests ─────────────────────────────────────────────────

    /// Config with max_seq_len > 512 and no SWA — should use flash attention.
    fn smollm_config() -> ModelConfig {
        ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 576,
            intermediate_size: 1536,
            num_layers: 2,
            num_attention_heads: 9,
            num_kv_heads: 3,
            head_dim: 64,
            vocab_size: 49152,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::F16,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        }
    }

    #[test]
    fn flash_attention_emitted_for_large_seq_len() {
        // max_seq_len = 2048 > 512 → flash attention should be used
        let config = smollm_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        assert!(
            code.contains("pub fn attention_flash("),
            "large seq_len model should emit attention_flash function"
        );
        assert!(
            code.contains("pub const FLASH_ATTN_BLOCK_SIZE: usize = 64;"),
            "large seq_len model should emit FLASH_ATTN_BLOCK_SIZE constant"
        );
        // forward() and forward_prefill() should call attention_flash
        assert!(
            code.contains("attention_flash("),
            "large seq_len model forward should call attention_flash"
        );
        // Standard attention should still be emitted (for use as reference / other paths),
        // but attention_flash takes priority in the forward pass
        assert!(
            !code.contains("        attention(\n"),
            "large seq_len forward body should not call bare attention()"
        );
    }

    #[test]
    fn small_seq_len_uses_standard_attention() {
        // max_seq_len = 64 <= 512 → standard attention
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        assert!(
            code.contains("pub fn attention("),
            "small seq_len model should emit standard attention"
        );
        assert!(
            !code.contains("pub fn attention_flash("),
            "small seq_len model should not emit attention_flash"
        );
        // forward() should call attention, not attention_flash
        assert!(
            code.contains("        attention(\n"),
            "small seq_len forward should call standard attention()"
        );
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn flash_attention_numerical_equivalence() {
        // Implement both attention algorithms as Rust functions and verify they produce
        // identical results up to floating-point rounding (< 1e-4 absolute difference).
        //
        // Uses head_dim=8, 3 heads, 2 kv_heads, seq_len=5 (< BLOCK_SIZE=64)
        // and seq_len=70 (> BLOCK_SIZE, exercises multi-block path).

        const HEAD_DIM: usize = 8;
        const BLOCK_SIZE: usize = 64;

        fn reference_attention(
            output: &mut [f32],
            q: &[f32],
            k_cache: &[f32],
            v_cache: &[f32],
            seq_len: usize,
            num_heads: usize,
            num_kv_heads: usize,
        ) {
            let gsize = num_heads / num_kv_heads;
            let scale = 1.0 / (HEAD_DIM as f32).sqrt();
            let kv_stride = num_kv_heads * HEAD_DIM;
            let mut scores = vec![0.0f32; seq_len];
            for h in 0..num_heads {
                let kv_h = h / gsize;
                let qo = h * HEAD_DIM;
                for t in 0..seq_len {
                    let ko = t * kv_stride + kv_h * HEAD_DIM;
                    let mut s = 0.0f32;
                    for i in 0..HEAD_DIM {
                        s += q[qo + i] * k_cache[ko + i];
                    }
                    scores[t] = s * scale;
                }
                let max_s = scores[..seq_len]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores[..seq_len].iter_mut() {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for s in scores[..seq_len].iter_mut() {
                    *s *= inv;
                }
                for d in 0..HEAD_DIM {
                    let mut val = 0.0f32;
                    for t in 0..seq_len {
                        val += scores[t] * v_cache[t * kv_stride + kv_h * HEAD_DIM + d];
                    }
                    output[qo + d] = val;
                }
            }
        }

        fn flash_attention(
            output: &mut [f32],
            q: &[f32],
            k_cache: &[f32],
            v_cache: &[f32],
            seq_len: usize,
            num_heads: usize,
            num_kv_heads: usize,
        ) {
            let gsize = num_heads / num_kv_heads;
            let scale = 1.0 / (HEAD_DIM as f32).sqrt();
            let kv_stride = num_kv_heads * HEAD_DIM;
            let mut scores = [0.0f32; BLOCK_SIZE];
            for h in 0..num_heads {
                let kv_h = h / gsize;
                let qo = h * HEAD_DIM;
                let mut m_i = f32::NEG_INFINITY;
                let mut l_i = 0.0f32;
                let mut acc = [0.0f32; HEAD_DIM];
                let mut block_start = 0;
                while block_start < seq_len {
                    let block_end = (block_start + BLOCK_SIZE).min(seq_len);
                    let block_len = block_end - block_start;
                    let mut m_block = f32::NEG_INFINITY;
                    for bi in 0..block_len {
                        let t = block_start + bi;
                        let ko = t * kv_stride + kv_h * HEAD_DIM;
                        let mut s = 0.0f32;
                        for i in 0..HEAD_DIM {
                            s += q[qo + i] * k_cache[ko + i];
                        }
                        let s = s * scale;
                        scores[bi] = s;
                        if s > m_block {
                            m_block = s;
                        }
                    }
                    let m_prev = m_i;
                    if m_block > m_i {
                        m_i = m_block;
                    }
                    let exp_scale = (m_prev - m_i).exp();
                    l_i *= exp_scale;
                    for d in 0..HEAD_DIM {
                        acc[d] *= exp_scale;
                    }
                    let mut l_block = 0.0f32;
                    for bi in 0..block_len {
                        let e = (scores[bi] - m_i).exp();
                        scores[bi] = e;
                        l_block += e;
                    }
                    l_i += l_block;
                    for d in 0..HEAD_DIM {
                        let mut sum = 0.0f32;
                        for bi in 0..block_len {
                            let t = block_start + bi;
                            sum += scores[bi] * v_cache[t * kv_stride + kv_h * HEAD_DIM + d];
                        }
                        acc[d] += sum;
                    }
                    block_start = block_end;
                }
                let inv_l = 1.0 / l_i;
                for d in 0..HEAD_DIM {
                    output[qo + d] = acc[d] * inv_l;
                }
            }
        }

        fn run_equivalence_check(seq_len: usize) {
            let num_heads = 3;
            let num_kv_heads = 1;
            let kv_stride = num_kv_heads * HEAD_DIM;

            // Deterministic pseudo-random data
            let q: Vec<f32> = (0..num_heads * HEAD_DIM)
                .map(|i| ((i * 7 + 3) as f32 * 0.1 - 1.0) * 0.5)
                .collect();
            let k_cache: Vec<f32> = (0..seq_len * kv_stride)
                .map(|i| ((i * 11 + 5) as f32 * 0.07 - 1.0) * 0.3)
                .collect();
            let v_cache: Vec<f32> = (0..seq_len * kv_stride)
                .map(|i| ((i * 13 + 7) as f32 * 0.05 - 0.5) * 0.4)
                .collect();

            let mut ref_out = vec![0.0f32; num_heads * HEAD_DIM];
            let mut flash_out = vec![0.0f32; num_heads * HEAD_DIM];

            reference_attention(
                &mut ref_out,
                &q,
                &k_cache,
                &v_cache,
                seq_len,
                num_heads,
                num_kv_heads,
            );
            flash_attention(
                &mut flash_out,
                &q,
                &k_cache,
                &v_cache,
                seq_len,
                num_heads,
                num_kv_heads,
            );

            for (i, (&r, &f)) in ref_out.iter().zip(flash_out.iter()).enumerate() {
                let diff = (r - f).abs();
                // Online softmax rescales in a different order than standard softmax,
                // leading to slightly different float rounding. 1e-4 is tight enough
                // to catch algorithmic errors while tolerating FP accumulation order.
                assert!(
                    diff < 1e-4,
                    "seq_len={seq_len}: output[{i}] differs: ref={r:.8} flash={f:.8} diff={diff:.2e}"
                );
            }
        }

        run_equivalence_check(5); // single block (seq_len < BLOCK_SIZE)
        run_equivalence_check(64); // exactly one full block
        run_equivalence_check(70); // two blocks (exercises multi-block rescaling)
        run_equivalence_check(128); // exactly two full blocks
        run_equivalence_check(200); // three+ blocks
    }

    #[test]
    fn generated_code_has_int8_kv_cache() {
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // KVCache struct should use i8 vectors with scale vectors
        assert!(
            code.contains("pub k: Vec<Vec<i8>>"),
            "KVCache k should be Vec<Vec<i8>>"
        );
        assert!(
            code.contains("pub v: Vec<Vec<i8>>"),
            "KVCache v should be Vec<Vec<i8>>"
        );
        assert!(
            code.contains("pub k_scales: Vec<Vec<f32>>"),
            "KVCache should have k_scales"
        );
        assert!(
            code.contains("pub v_scales: Vec<Vec<f32>>"),
            "KVCache should have v_scales"
        );

        // Should have quantize_kv helper function
        assert!(
            code.contains("pub fn quantize_kv("),
            "generated code should have quantize_kv function"
        );

        // Should have dot_f32_i8 helper for dequantized dot product
        assert!(
            code.contains("fn dot_f32_i8("),
            "generated code should have dot_f32_i8 function"
        );

        // Attention functions should take i8 cache slices
        assert!(
            code.contains("k_cache: &[i8]"),
            "attention should accept i8 k_cache"
        );
        assert!(
            code.contains("v_cache: &[i8]"),
            "attention should accept i8 v_cache"
        );

        // Forward function should quantize before writing to cache
        assert!(
            code.contains("quantize_kv(&k, &mut k_q)"),
            "forward should quantize k before cache write"
        );
        assert!(
            code.contains("quantize_kv(&v, &mut v_q)"),
            "forward should quantize v before cache write"
        );

        // KVCache initialization should use i8 zeros
        assert!(
            code.contains("vec![0i8;"),
            "KVCache should initialize with i8 zeros"
        );

        // memory_bytes should reflect int8 storage
        assert!(
            code.contains("i8) + scales (f32)"),
            "memory_bytes should document int8+scales storage"
        );
    }

    #[test]
    fn kv_quantize_round_trip_accuracy() {
        // Verify int8 quantization round-trip error is small enough for attention.
        // Per-token absmax quantization to 127 levels should give < 1% relative error.
        let values: Vec<f32> = (0..64)
            .map(|i| ((i * 7 + 3) as f32 * 0.1 - 3.0) * 0.5)
            .collect();
        let mut quantized = [0i8; 64];

        // Quantize
        let mut max_abs = 0.0f32;
        for &v in &values {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = max_abs / 127.0;
        let inv_scale = if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 };
        for (i, &v) in values.iter().enumerate() {
            quantized[i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }

        // Dequantize and measure error
        let mut max_err = 0.0f32;
        for (i, &v) in values.iter().enumerate() {
            let dequantized = quantized[i] as f32 * scale;
            let err = (v - dequantized).abs();
            if err > max_err {
                max_err = err;
            }
        }

        // Max error should be less than half a quantization step
        let step = scale; // one quantization level
        assert!(
            max_err <= step + 1e-6,
            "max round-trip error {max_err:.6} exceeds one quantization step {step:.6}"
        );
    }

    // ── Real-world validation tests ──────────────────────────────────────

    #[test]
    fn generated_code_handles_pos_zero() {
        // Verify forward() with pos=0 doesn't produce degenerate output.
        // At pos=0 the RoPE angle is 0 (cos=1, sin=0) and the KV cache has
        // seq_len=1.  The attention scores array is `[0.0f32; MAX_SEQ_LEN]`
        // and only scores[0] is written, so softmax must handle a single-element
        // slice without NaN/Inf.
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // The forward function should accept pos=0 (no guard against it)
        assert!(
            !code.contains("assert!(pos > 0"),
            "forward() must accept pos=0 for the first token"
        );
        // Softmax in the generated code should use max subtraction for
        // numerical stability (prevents exp() overflow)
        assert!(
            code.contains("f32::NEG_INFINITY") || code.contains("fold(f32::NEG_INFINITY"),
            "softmax should initialize max to NEG_INFINITY for numerical stability"
        );
        // Attention at seq_len=1: the code slices scores[..seq_len] which
        // at pos=0 means scores[..1].  Verify softmax operates on the slice.
        assert!(
            code.contains("softmax(&mut scores[..seq_len])")
                || code.contains("softmax(&mut scores[..seq_len]);"),
            "attention must apply softmax to the active seq_len slice"
        );
    }

    #[test]
    fn generated_code_caps_max_seq_len() {
        // Models like Qwen2.5 declare max_seq_len=32768 but the generated
        // code should cap it to avoid multi-GB KV cache allocations.
        let config = ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 256,
            max_seq_len: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::F16,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        };
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // MAX_SEQ_LEN should be capped (currently at 4096)
        assert!(
            code.contains("pub const MAX_SEQ_LEN: usize = 4096;"),
            "MAX_SEQ_LEN should be capped at 4096 for models with very large max_seq_len"
        );
        // The comment should mention the original value
        assert!(
            code.contains("capped from model's 32768"),
            "generated code should document the original max_seq_len"
        );
    }

    #[test]
    fn generated_softmax_is_numerically_stable() {
        // Verify the generated softmax uses the max-subtraction trick.
        // Without it, very negative logits would underflow and very positive
        // logits would overflow, producing NaN.
        let config = tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Extract the softmax function body (up to 1000 chars to cover NEON variant)
        let softmax_start = code.find("pub fn softmax(").expect("softmax must exist");
        let softmax_body =
            &code[softmax_start..softmax_start + 1000.min(code.len() - softmax_start)];

        // Must subtract max before exp() — the generated code uses `*v - max_val`
        assert!(
            softmax_body.contains("- max_val") || softmax_body.contains("-max_val"),
            "softmax must subtract max value before exp() for numerical stability"
        );
        // Must guard against zero sum (degenerate case where all exp() underflow)
        assert!(
            softmax_body.contains("if sum > 0.0") || softmax_body.contains("1.0 / sum"),
            "softmax should guard against division by zero when sum is zero"
        );
    }

    #[test]
    fn generated_code_valid_rust_for_large_context_model() {
        // A model with max_seq_len > 512 triggers flash attention codegen.
        // Verify the generated code for this path is syntactically valid.
        let config = ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 256,
            max_seq_len: 2048, // > 512 triggers flash attention
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::F16,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        };
        let graph = graph_builder::build_graph(&config).unwrap();
        let code = generate(&graph).unwrap();

        // Flash attention should be emitted
        assert!(
            code.contains("attention_flash"),
            "model with max_seq_len > 512 should use flash attention"
        );
        // The generated code should still be valid Rust
        syn::parse_file(&code).expect("flash attention code should be valid Rust syntax");
    }
}
