//! Optimized compute kernels.
//!
//! Provides SIMD-accelerated implementations of core operations.
//! Uses ARM NEON intrinsics on aarch64, falls back to scalar code elsewhere.
//!
//! The primary bottleneck is matmul (matrix-vector multiply for single-token
//! generation). The NEON version processes 4 f32s per cycle using vfmaq_f32.

/// NEON-accelerated dot product of two f32 slices.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_f32_neon(a: &[f32], b: &[f32], len: usize) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let chunks = len / 16;
        for i in 0..chunks {
            let base = i * 16;
            let a0 = vld1q_f32(a.as_ptr().add(base));
            let b0 = vld1q_f32(b.as_ptr().add(base));
            sum0 = vfmaq_f32(sum0, a0, b0);

            let a1 = vld1q_f32(a.as_ptr().add(base + 4));
            let b1 = vld1q_f32(b.as_ptr().add(base + 4));
            sum1 = vfmaq_f32(sum1, a1, b1);

            let a2 = vld1q_f32(a.as_ptr().add(base + 8));
            let b2 = vld1q_f32(b.as_ptr().add(base + 8));
            sum2 = vfmaq_f32(sum2, a2, b2);

            let a3 = vld1q_f32(a.as_ptr().add(base + 12));
            let b3 = vld1q_f32(b.as_ptr().add(base + 12));
            sum3 = vfmaq_f32(sum3, a3, b3);
        }

        // Combine accumulators
        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);

        let mut result = vaddvq_f32(sum0);

        // Handle remainder
        for i in (chunks * 16)..len {
            result += *a.get_unchecked(i) * *b.get_unchecked(i);
        }

        result
    }
}

/// Scalar dot product fallback.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot_f32_neon(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Optimized matrix-vector multiply using NEON dot products.
///
/// For single-token inference (m=1), computes dot products between
/// the input vector and each weight row.
///
/// Weight layout: [n, k] (row-major), so weight row j is at offset j*k.
pub fn matmul_vec(output: &mut [f32], input: &[f32], weight: &[f32], k: usize, n: usize) {
    // Process 4 output rows at a time for ILP
    let n_chunks = n / 4;
    let n_remainder = n % 4;

    for chunk in 0..n_chunks {
        let j0 = chunk * 4;
        output[j0] = dot_f32_neon(input, &weight[j0 * k..(j0 + 1) * k], k);
        output[j0 + 1] = dot_f32_neon(input, &weight[(j0 + 1) * k..(j0 + 2) * k], k);
        output[j0 + 2] = dot_f32_neon(input, &weight[(j0 + 2) * k..(j0 + 3) * k], k);
        output[j0 + 3] = dot_f32_neon(input, &weight[(j0 + 3) * k..(j0 + 4) * k], k);
    }

    // Handle remaining output elements
    let j_base = n_chunks * 4;
    for r in 0..n_remainder {
        let j = j_base + r;
        output[j] = dot_f32_neon(input, &weight[j * k..(j + 1) * k], k);
    }
}

/// General matrix multiply: output[m,n] = input[m,k] * weight^T[k,n]
///
/// For m=1 (single token), delegates to the optimized vector version.
pub fn matmul(output: &mut [f32], input: &[f32], weight: &[f32], m: usize, k: usize, n: usize) {
    if m == 1 {
        matmul_vec(output, input, weight, k, n);
    } else {
        for i in 0..m {
            let in_row = &input[i * k..(i + 1) * k];
            let out_row = &mut output[i * n..(i + 1) * n];
            matmul_vec(out_row, in_row, weight, k, n);
        }
    }
}

/// NEON-accelerated RMSNorm: output = (input / rms(input)) * weight
pub fn rms_norm(output: &mut [f32], input: &[f32], weight: &[f32], eps: f32) {
    let n = input.len();

    // NEON dot product for sum of squares
    let sum_sq = dot_f32_neon(input, input, n);
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    // NEON-accelerated normalization + weight multiply
    rms_norm_apply(output, input, weight, inv_rms);
}

#[cfg(target_arch = "aarch64")]
fn rms_norm_apply(output: &mut [f32], input: &[f32], weight: &[f32], inv_rms: f32) {
    use std::arch::aarch64::*;
    let n = input.len();
    let chunks = n / 4;

    unsafe {
        let scale = vdupq_n_f32(inv_rms);
        for i in 0..chunks {
            let base = i * 4;
            let x = vld1q_f32(input.as_ptr().add(base));
            let w = vld1q_f32(weight.as_ptr().add(base));
            let r = vmulq_f32(vmulq_f32(x, scale), w);
            vst1q_f32(output.as_mut_ptr().add(base), r);
        }
    }
    for i in (chunks * 4)..n {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn rms_norm_apply(output: &mut [f32], input: &[f32], weight: &[f32], inv_rms: f32) {
    for i in 0..input.len() {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

/// Optimized SiLU: x * sigmoid(x) = x / (1 + exp(-x))
pub fn silu(output: &mut [f32], input: &[f32]) {
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x / (1.0 + (-x).exp());
    }
}

/// GeLU activation (approximate): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(output: &mut [f32], input: &[f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/pi)
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        let inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
        *o = 0.5 * x * (1.0 + inner.tanh());
    }
}

/// NEON-accelerated elementwise multiply
pub fn elementwise_mul(output: &mut [f32], a: &[f32], b: &[f32]) {
    elementwise_binary_op(output, a, b, BinaryOp::Mul);
}

/// NEON-accelerated elementwise add
pub fn elementwise_add(output: &mut [f32], a: &[f32], b: &[f32]) {
    elementwise_binary_op(output, a, b, BinaryOp::Add);
}

enum BinaryOp {
    Mul,
    Add,
}

#[cfg(target_arch = "aarch64")]
fn elementwise_binary_op(output: &mut [f32], a: &[f32], b: &[f32], op: BinaryOp) {
    use std::arch::aarch64::*;
    let n = a.len();
    let chunks = n / 16;

    unsafe {
        for i in 0..chunks {
            let base = i * 16;
            let a0 = vld1q_f32(a.as_ptr().add(base));
            let b0 = vld1q_f32(b.as_ptr().add(base));
            let a1 = vld1q_f32(a.as_ptr().add(base + 4));
            let b1 = vld1q_f32(b.as_ptr().add(base + 4));
            let a2 = vld1q_f32(a.as_ptr().add(base + 8));
            let b2 = vld1q_f32(b.as_ptr().add(base + 8));
            let a3 = vld1q_f32(a.as_ptr().add(base + 12));
            let b3 = vld1q_f32(b.as_ptr().add(base + 12));

            let (r0, r1, r2, r3) = match op {
                BinaryOp::Mul => (
                    vmulq_f32(a0, b0),
                    vmulq_f32(a1, b1),
                    vmulq_f32(a2, b2),
                    vmulq_f32(a3, b3),
                ),
                BinaryOp::Add => (
                    vaddq_f32(a0, b0),
                    vaddq_f32(a1, b1),
                    vaddq_f32(a2, b2),
                    vaddq_f32(a3, b3),
                ),
            };

            vst1q_f32(output.as_mut_ptr().add(base), r0);
            vst1q_f32(output.as_mut_ptr().add(base + 4), r1);
            vst1q_f32(output.as_mut_ptr().add(base + 8), r2);
            vst1q_f32(output.as_mut_ptr().add(base + 12), r3);
        }
    }

    // Scalar remainder
    for i in (chunks * 16)..n {
        output[i] = match op {
            BinaryOp::Mul => a[i] * b[i],
            BinaryOp::Add => a[i] + b[i],
        };
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn elementwise_binary_op(output: &mut [f32], a: &[f32], b: &[f32], op: BinaryOp) {
    for i in 0..a.len() {
        output[i] = match op {
            BinaryOp::Mul => a[i] * b[i],
            BinaryOp::Add => a[i] + b[i],
        };
    }
}

/// Softmax with numerical stability
pub fn softmax(values: &mut [f32]) {
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum: f32 = 0.0;
    for v in values.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in values.iter_mut() {
        *v *= inv_sum;
    }
}

/// Optimized grouped-query attention using NEON dot products.
///
/// Computes: for each head, score = softmax(Q·K^T / sqrt(d)), output = score·V
#[allow(clippy::too_many_arguments)]
pub fn attention(
    output: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) {
    let kv_group_size = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_stride = num_kv_heads * head_dim;

    for h in 0..num_heads {
        let kv_h = h / kv_group_size;
        let q_offset = h * head_dim;
        let q_head = &q[q_offset..q_offset + head_dim];

        // Compute attention scores using NEON dot product
        let mut scores = vec![0.0f32; seq_len];
        for (t, score) in scores.iter_mut().enumerate() {
            let k_offset = t * kv_stride + kv_h * head_dim;
            *score =
                dot_f32_neon(q_head, &k_cache[k_offset..k_offset + head_dim], head_dim) * scale;
        }

        softmax(&mut scores);

        // Weighted sum of values
        for d in 0..head_dim {
            let mut sum: f32 = 0.0;
            for (t, &score) in scores.iter().enumerate() {
                let v_offset = t * kv_stride + kv_h * head_dim;
                sum += score * v_cache[v_offset + d];
            }
            output[q_offset + d] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_basic() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 1.0, 1.0, 1.0];
        let result = dot_f32_neon(&a, &b, 4);
        assert!((result - 10.0).abs() < 1e-5);
    }

    #[test]
    fn dot_product_large() {
        let k = 576; // SmolLM hidden size
        let a: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k).map(|i| ((k - i) as f32) * 0.001).collect();

        let neon_result = dot_f32_neon(&a, &b, k);

        // Reference
        let ref_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (neon_result - ref_result).abs() < 1e-1,
            "NEON={neon_result}, ref={ref_result}"
        );
    }

    #[test]
    fn matmul_vec_basic() {
        // [1, 2] * [[1, 2], [3, 4]]^T = [1*1+2*2, 1*3+2*4] = [5, 11]
        let input = [1.0f32, 2.0];
        let weight = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 2];
        matmul_vec(&mut output, &input, &weight, 2, 2);
        assert!((output[0] - 5.0).abs() < 1e-5);
        assert!((output[1] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_vec_larger() {
        let k = 64;
        let n = 32;
        let input: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
        let weight: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.01).collect();
        let mut output = vec![0.0f32; n];
        let mut output_ref = vec![0.0f32; n];

        matmul_vec(&mut output, &input, &weight, k, n);

        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += input[l] * weight[j * k + l];
            }
            output_ref[j] = sum;
        }

        for j in 0..n {
            assert!(
                (output[j] - output_ref[j]).abs() < 1e-2,
                "mismatch at j={j}: {} vs {}",
                output[j],
                output_ref[j]
            );
        }
    }

    #[test]
    fn matmul_vec_odd_dimensions() {
        let k = 13;
        let n = 7;
        let input: Vec<f32> = (0..k).map(|i| i as f32).collect();
        let weight: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.01).collect();
        let mut output = vec![0.0f32; n];
        let mut output_ref = vec![0.0f32; n];

        matmul_vec(&mut output, &input, &weight, k, n);

        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += input[l] * weight[j * k + l];
            }
            output_ref[j] = sum;
        }

        for j in 0..n {
            assert!(
                (output[j] - output_ref[j]).abs() < 1e-2,
                "mismatch at j={j}: {} vs {}",
                output[j],
                output_ref[j]
            );
        }
    }

    #[test]
    fn rms_norm_basic() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let weight = [1.0f32; 4];
        let mut output = [0.0f32; 4];
        let mut output_ref = [0.0f32; 4];

        rms_norm(&mut output, &input, &weight, 1e-5);

        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (sum_sq / 4.0 + 1e-5).sqrt();
        for i in 0..4 {
            output_ref[i] = input[i] * inv_rms * weight[i];
        }

        for i in 0..4 {
            assert!((output[i] - output_ref[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn matmul_general() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let weight = [1.0, 0.0, 0.0, 1.0];
        let mut output = [0.0f32; 4];
        matmul(&mut output, &input, &weight, 2, 2, 2);
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
        assert!((output[2] - 3.0).abs() < 1e-5);
        assert!((output[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn dot_product_smollm_dimension() {
        // Test with actual SmolLM hidden dimension (576)
        let k = 576;
        let a: Vec<f32> = (0..k).map(|i| ((i * 7 + 3) % 100) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..k).map(|i| ((i * 11 + 5) % 100) as f32 * 0.01).collect();

        let neon = dot_f32_neon(&a, &b, k);
        let reference: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!(
            (neon - reference).abs() / reference.abs() < 1e-4,
            "relative error too large: NEON={neon}, ref={reference}"
        );
    }
}
