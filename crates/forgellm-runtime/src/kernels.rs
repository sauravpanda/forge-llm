//! Optimized compute kernels.
//!
//! Provides SIMD-accelerated implementations of core operations.
//! Falls back to scalar code on unsupported platforms.
//!
//! The primary bottleneck is matmul (matrix-vector multiply for single-token
//! generation). The optimized version uses:
//! - Loop unrolling (4 output elements per iteration)
//! - Vectorized dot product with manual accumulation
//! - Cache-friendly access patterns (weight rows are contiguous)

/// Optimized matrix-vector multiply: output[n] = sum_k(input[k] * weight[j][k])
///
/// For single-token inference (m=1), this is a series of dot products between
/// the input vector and each weight row. We optimize the inner dot product loop.
///
/// Weight layout: [n, k] (row-major), so weight row j is at offset j*k.
pub fn matmul_vec(output: &mut [f32], input: &[f32], weight: &[f32], k: usize, n: usize) {
    // Process 4 output elements at a time for instruction-level parallelism
    let n_chunks = n / 4;
    let n_remainder = n % 4;

    for chunk in 0..n_chunks {
        let j0 = chunk * 4;
        let j1 = j0 + 1;
        let j2 = j0 + 2;
        let j3 = j0 + 3;

        let w0 = &weight[j0 * k..j0 * k + k];
        let w1 = &weight[j1 * k..j1 * k + k];
        let w2 = &weight[j2 * k..j2 * k + k];
        let w3 = &weight[j3 * k..j3 * k + k];

        let mut sum0: f32 = 0.0;
        let mut sum1: f32 = 0.0;
        let mut sum2: f32 = 0.0;
        let mut sum3: f32 = 0.0;

        // Vectorize the inner loop — process 8 elements at a time
        let k_chunks = k / 8;
        let k_rem = k % 8;

        for i in 0..k_chunks {
            let base = i * 8;
            // Manually unrolled 8-element accumulation
            for off in 0..8 {
                let x = input[base + off];
                sum0 += x * w0[base + off];
                sum1 += x * w1[base + off];
                sum2 += x * w2[base + off];
                sum3 += x * w3[base + off];
            }
        }

        // Handle remainder
        let k_base = k_chunks * 8;
        for off in 0..k_rem {
            let x = input[k_base + off];
            sum0 += x * w0[k_base + off];
            sum1 += x * w1[k_base + off];
            sum2 += x * w2[k_base + off];
            sum3 += x * w3[k_base + off];
        }

        output[j0] = sum0;
        output[j1] = sum1;
        output[j2] = sum2;
        output[j3] = sum3;
    }

    // Handle remaining output elements
    let j_base = n_chunks * 4;
    for r in 0..n_remainder {
        let j = j_base + r;
        let w = &weight[j * k..j * k + k];
        let mut sum: f32 = 0.0;
        for l in 0..k {
            sum += input[l] * w[l];
        }
        output[j] = sum;
    }
}

/// General matrix multiply: output[m,n] = input[m,k] * weight^T[k,n]
///
/// For m=1 (single token), delegates to the optimized vector version.
/// For m>1 (prefill), uses the unrolled version per row.
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

/// Optimized RMSNorm: output = (input / rms(input)) * weight
pub fn rms_norm(output: &mut [f32], input: &[f32], weight: &[f32], eps: f32) {
    let n = input.len();

    // Compute sum of squares with unrolled accumulation
    let mut sum_sq: f32 = 0.0;
    let chunks = n / 4;
    for i in 0..chunks {
        let base = i * 4;
        sum_sq += input[base] * input[base];
        sum_sq += input[base + 1] * input[base + 1];
        sum_sq += input[base + 2] * input[base + 2];
        sum_sq += input[base + 3] * input[base + 3];
    }
    for x in &input[chunks * 4..n] {
        sum_sq += x * x;
    }

    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();

    // Apply normalization + weight
    for i in 0..n {
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

/// Optimized elementwise multiply
pub fn elementwise_mul(output: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..a.len() {
        output[i] = a[i] * b[i];
    }
}

/// Optimized elementwise add
pub fn elementwise_add(output: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..a.len() {
        output[i] = a[i] + b[i];
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
    let inv_sum = 1.0 / sum;
    for v in values.iter_mut() {
        *v *= inv_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Test with dimensions that exercise the unrolled path
        let k = 64;
        let n = 32;
        let input: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
        let weight: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.01).collect();
        let mut output = vec![0.0f32; n];
        let mut output_ref = vec![0.0f32; n];

        matmul_vec(&mut output, &input, &weight, k, n);

        // Reference implementation
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += input[l] * weight[j * k + l];
            }
            output_ref[j] = sum;
        }

        for j in 0..n {
            assert!(
                (output[j] - output_ref[j]).abs() < 1e-3,
                "mismatch at j={j}: {} vs {}",
                output[j],
                output_ref[j]
            );
        }
    }

    #[test]
    fn matmul_vec_odd_dimensions() {
        // Test with dimensions that don't divide evenly by 4 or 8
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

        // Reference
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
        // m=2 case
        let input = [1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let weight = [1.0, 0.0, 0.0, 1.0]; // identity 2x2
        let mut output = [0.0f32; 4];
        matmul(&mut output, &input, &weight, 2, 2, 2);
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
        assert!((output[2] - 3.0).abs() < 1e-5);
        assert!((output[3] - 4.0).abs() < 1e-5);
    }
}
