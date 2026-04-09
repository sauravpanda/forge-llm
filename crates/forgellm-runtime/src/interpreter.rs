//! Reference interpreter — executes IR graphs directly on CPU.
//!
//! This is a straightforward, unoptimized interpreter that runs each
//! IR operation in sequence using f32 arithmetic. Used to validate
//! correctness before the compiled codegen path.

use forgellm_frontend::ir::*;
use forgellm_frontend::weight_loader::ModelWeights;

use crate::kv_cache::KVCache;

/// Run a single forward pass for one token through the model.
///
/// Returns logits of shape `[vocab_size]`.
pub fn forward(
    token_id: u32,
    pos: usize,
    graph: &Graph,
    weights: &ModelWeights,
    cache: &mut KVCache,
) -> Vec<f32> {
    let config = graph.config.as_ref().expect("graph must have config");

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let vocab = config.vocab_size;

    // Embedding lookup
    let embed_w = weights.tensor("model.embed_tokens.weight");
    let mut hidden_state = vec![0.0f32; hidden];
    let offset = token_id as usize * hidden;
    hidden_state.copy_from_slice(&embed_w[offset..offset + hidden]);

    // Pre-allocate buffers
    let mut normed = vec![0.0f32; hidden];
    let mut q = vec![0.0f32; num_heads * head_dim];
    let mut k = vec![0.0f32; num_kv_heads * head_dim];
    let mut v = vec![0.0f32; num_kv_heads * head_dim];
    let mut attn_out = vec![0.0f32; num_heads * head_dim];
    let mut attn_proj = vec![0.0f32; hidden];
    let mut residual = vec![0.0f32; hidden];
    let mut gate = vec![0.0f32; intermediate];
    let mut gate_act = vec![0.0f32; intermediate];
    let mut up = vec![0.0f32; intermediate];
    let mut ffn_hidden = vec![0.0f32; intermediate];
    let mut ffn_out = vec![0.0f32; hidden];

    for layer_idx in 0..config.num_layers {
        let prefix = format!("model.layers.{layer_idx}");

        // Attention norm
        let norm_w = weights.tensor(&format!("{prefix}.input_layernorm.weight"));
        rms_norm(&mut normed, &hidden_state, norm_w, config.rms_norm_eps);

        // QKV projections
        let q_w = weights.tensor(&format!("{prefix}.self_attn.q_proj.weight"));
        let k_w = weights.tensor(&format!("{prefix}.self_attn.k_proj.weight"));
        let v_w = weights.tensor(&format!("{prefix}.self_attn.v_proj.weight"));
        matmul(&mut q, &normed, q_w, 1, hidden, num_heads * head_dim);
        matmul(&mut k, &normed, k_w, 1, hidden, num_kv_heads * head_dim);
        matmul(&mut v, &normed, v_w, 1, hidden, num_kv_heads * head_dim);

        // RoPE
        rope(&mut q, pos, head_dim, num_heads, config.rope_theta);
        rope(&mut k, pos, head_dim, num_kv_heads, config.rope_theta);

        // Update KV cache
        cache.append(layer_idx, &k, &v);

        // Attention
        attention(
            &mut attn_out,
            &q,
            cache.k(layer_idx),
            cache.v(layer_idx),
            &AttentionParams {
                seq_len: pos + 1,
                num_heads,
                num_kv_heads,
                head_dim,
            },
        );

        // Output projection
        let o_w = weights.tensor(&format!("{prefix}.self_attn.o_proj.weight"));
        matmul(
            &mut attn_proj,
            &attn_out,
            o_w,
            1,
            num_heads * head_dim,
            hidden,
        );

        // Residual
        elementwise_add(&mut residual, &hidden_state, &attn_proj);

        // FFN norm
        let ffn_norm_w = weights.tensor(&format!("{prefix}.post_attention_layernorm.weight"));
        rms_norm(&mut normed, &residual, ffn_norm_w, config.rms_norm_eps);

        // FFN
        let gate_w = weights.tensor(&format!("{prefix}.mlp.gate_proj.weight"));
        let up_w = weights.tensor(&format!("{prefix}.mlp.up_proj.weight"));
        let down_w = weights.tensor(&format!("{prefix}.mlp.down_proj.weight"));

        matmul(&mut gate, &normed, gate_w, 1, hidden, intermediate);
        silu(&mut gate_act, &gate);
        matmul(&mut up, &normed, up_w, 1, hidden, intermediate);
        elementwise_mul(&mut ffn_hidden, &gate_act, &up);
        matmul(&mut ffn_out, &ffn_hidden, down_w, 1, intermediate, hidden);

        // Residual
        elementwise_add(&mut hidden_state, &residual, &ffn_out);
    }

    // Final norm
    let final_norm_w = weights.tensor("model.norm.weight");
    rms_norm(
        &mut normed,
        &hidden_state,
        final_norm_w,
        config.rms_norm_eps,
    );

    // Logits projection (may use tied embeddings)
    let lm_head_w = weights
        .get("lm_head.weight")
        .unwrap_or_else(|| weights.tensor("model.embed_tokens.weight"));
    let mut logits = vec![0.0f32; vocab];
    matmul(&mut logits, &normed, lm_head_w, 1, hidden, vocab);

    logits
}

// --- Kernel implementations ---

fn rms_norm(output: &mut [f32], input: &[f32], weight: &[f32], eps: f32) {
    let n = input.len();
    let mut sum_sq: f32 = 0.0;
    for &x in input {
        sum_sq += x * x;
    }
    let inv_rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    for i in 0..n {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

/// Matrix multiply: output[m,n] = input[m,k] * weight^T[k,n]
/// Weight layout: [n, k] (row-major).
fn matmul(output: &mut [f32], input: &[f32], weight: &[f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum: f32 = 0.0;
            for l in 0..k {
                sum += input[i * k + l] * weight[j * k + l];
            }
            output[i * n + j] = sum;
        }
    }
}

fn silu(output: &mut [f32], input: &[f32]) {
    for i in 0..input.len() {
        let x = input[i];
        output[i] = x / (1.0 + (-x).exp());
    }
}

fn elementwise_mul(output: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..a.len() {
        output[i] = a[i] * b[i];
    }
}

fn elementwise_add(output: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..a.len() {
        output[i] = a[i] + b[i];
    }
}

fn softmax(values: &mut [f32]) {
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum: f32 = 0.0;
    for v in values.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    for v in values.iter_mut() {
        *v /= sum;
    }
}

fn rope(data: &mut [f32], pos: usize, head_dim: usize, num_heads: usize, theta: f32) {
    for h in 0..num_heads {
        let head_offset = h * head_dim;
        for i in (0..head_dim).step_by(2) {
            let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos_val = angle.cos();
            let sin_val = angle.sin();
            let x0 = data[head_offset + i];
            let x1 = data[head_offset + i + 1];
            data[head_offset + i] = x0 * cos_val - x1 * sin_val;
            data[head_offset + i + 1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

struct AttentionParams {
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

fn attention(
    output: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    params: &AttentionParams,
) {
    let kv_group_size = params.num_heads / params.num_kv_heads;
    let scale = 1.0 / (params.head_dim as f32).sqrt();

    for h in 0..params.num_heads {
        let kv_h = h / kv_group_size;
        let q_offset = h * params.head_dim;

        // Compute attention scores
        let mut scores = vec![0.0f32; params.seq_len];
        for (t, score) in scores.iter_mut().enumerate() {
            let k_offset = t * params.num_kv_heads * params.head_dim + kv_h * params.head_dim;
            let mut dot: f32 = 0.0;
            for d in 0..params.head_dim {
                dot += q[q_offset + d] * k_cache[k_offset + d];
            }
            *score = dot * scale;
        }

        // Softmax
        softmax(&mut scores);

        // Weighted sum of values
        for d in 0..params.head_dim {
            let mut sum: f32 = 0.0;
            for (t, &score) in scores.iter().enumerate() {
                let v_offset = t * params.num_kv_heads * params.head_dim + kv_h * params.head_dim;
                sum += score * v_cache[v_offset + d];
            }
            output[q_offset + d] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn rms_norm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        rms_norm(&mut output, &input, &weight, 1e-5);

        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (30.0f32 / 4.0 + 1e-5).sqrt();
        let expected: Vec<f32> = input.iter().map(|x| x / rms).collect();
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {a}, expected {b}");
        }
    }

    #[test]
    fn matmul_basic() {
        // [1, 2] x [[1, 3], [2, 4]]^T = [1*1+2*2, 1*3+2*4] = [5, 11]
        // weight stored as [n, k] = [[1, 2], [3, 4]]
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // row 0: [1,2], row 1: [3,4]
        let mut output = vec![0.0; 2];
        matmul(&mut output, &input, &weight, 1, 2, 2);
        assert!((output[0] - 5.0).abs() < 1e-6);
        assert!((output[1] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn silu_basic() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0; 3];
        silu(&mut output, &input);
        // silu(0) = 0, silu(1) = 1/(1+e^-1) ≈ 0.7311
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1] - 0.7311).abs() < 1e-3);
        assert!((output[2] - (-0.2689)).abs() < 1e-3);
    }

    #[test]
    fn softmax_basic() {
        let mut values = vec![1.0, 2.0, 3.0];
        softmax(&mut values);
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(values[2] > values[1]);
        assert!(values[1] > values[0]);
    }

    #[test]
    fn rope_preserves_magnitude() {
        // RoPE is a rotation, so it should preserve vector magnitude
        let mut data = vec![1.0, 0.0, 0.0, 1.0]; // 1 head, dim=4
        let mag_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        rope(&mut data, 5, 4, 1, 10000.0);
        let mag_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (mag_before - mag_after).abs() < 1e-5,
            "RoPE changed magnitude: {mag_before} → {mag_after}"
        );
    }

    #[test]
    fn forward_with_tiny_model() {
        // Build a minimal model to verify the interpreter runs without panicking
        let config = ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 8,
            intermediate_size: 16,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            vocab_size: 16,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::F32,
        };

        let graph = forgellm_frontend::graph_builder::build_graph(&config).unwrap();

        // Create random-ish weights
        let mut tensors = HashMap::new();
        let h = 8;
        let inter = 16;
        let vocab = 16;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;

        tensors.insert("model.embed_tokens.weight".into(), vec![0.1f32; vocab * h]);
        tensors.insert(
            "model.layers.0.input_layernorm.weight".into(),
            vec![1.0f32; h],
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            vec![0.01f32; num_heads * head_dim * h],
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".into(),
            vec![0.01f32; num_kv_heads * head_dim * h],
        );
        tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".into(),
            vec![0.01f32; num_kv_heads * head_dim * h],
        );
        tensors.insert(
            "model.layers.0.self_attn.o_proj.weight".into(),
            vec![0.01f32; h * num_heads * head_dim],
        );
        tensors.insert(
            "model.layers.0.post_attention_layernorm.weight".into(),
            vec![1.0f32; h],
        );
        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".into(),
            vec![0.01f32; inter * h],
        );
        tensors.insert(
            "model.layers.0.mlp.up_proj.weight".into(),
            vec![0.01f32; inter * h],
        );
        tensors.insert(
            "model.layers.0.mlp.down_proj.weight".into(),
            vec![0.01f32; h * inter],
        );
        tensors.insert("model.norm.weight".into(), vec![1.0f32; h]);
        tensors.insert("lm_head.weight".into(), vec![0.01f32; vocab * h]);

        let weights = ModelWeights { tensors };
        let mut kv_cache = KVCache::new(1, num_kv_heads, head_dim);

        // Run forward pass
        let logits = forward(0, 0, &graph, &weights, &mut kv_cache);
        assert_eq!(logits.len(), vocab);
        assert_eq!(kv_cache.len(), 0); // advance not called by forward

        // Logits should be finite
        for &l in &logits {
            assert!(l.is_finite(), "logit is not finite: {l}");
        }
    }

    #[test]
    fn forward_multi_token() {
        let config = ModelConfig {
            architecture: Architecture::Llama,
            hidden_size: 8,
            intermediate_size: 16,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            vocab_size: 16,
            max_seq_len: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::F32,
        };

        let graph = forgellm_frontend::graph_builder::build_graph(&config).unwrap();

        let mut tensors = HashMap::new();
        let h = 8;
        let inter = 16;
        let vocab = 16;

        tensors.insert("model.embed_tokens.weight".into(), vec![0.1f32; vocab * h]);
        tensors.insert("model.layers.0.input_layernorm.weight".into(), vec![1.0; h]);
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".into(),
            vec![0.01; 8 * h],
        );
        tensors.insert(
            "model.layers.0.self_attn.k_proj.weight".into(),
            vec![0.01; 4 * h],
        );
        tensors.insert(
            "model.layers.0.self_attn.v_proj.weight".into(),
            vec![0.01; 4 * h],
        );
        tensors.insert(
            "model.layers.0.self_attn.o_proj.weight".into(),
            vec![0.01; h * 8],
        );
        tensors.insert(
            "model.layers.0.post_attention_layernorm.weight".into(),
            vec![1.0; h],
        );
        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".into(),
            vec![0.01; inter * h],
        );
        tensors.insert(
            "model.layers.0.mlp.up_proj.weight".into(),
            vec![0.01; inter * h],
        );
        tensors.insert(
            "model.layers.0.mlp.down_proj.weight".into(),
            vec![0.01; h * inter],
        );
        tensors.insert("model.norm.weight".into(), vec![1.0; h]);
        tensors.insert("lm_head.weight".into(), vec![0.01; vocab * h]);

        let weights = ModelWeights { tensors };
        let mut cache = KVCache::new(1, 1, 4);

        // Generate 3 tokens
        for pos in 0..3 {
            let logits = forward(1, pos, &graph, &weights, &mut cache);
            assert_eq!(logits.len(), vocab);
            cache.advance();
        }

        assert_eq!(cache.len(), 3);
    }
}
