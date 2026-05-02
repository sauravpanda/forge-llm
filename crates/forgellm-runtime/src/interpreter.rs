//! Interpreter — executes IR graphs directly on CPU.
//!
//! Uses optimized kernels from the `kernels` module for compute-heavy
//! operations (matmul, rms_norm). Validates correctness and serves as
//! the primary inference path until the AOT codegen is ready.

use forgellm_frontend::ir::*;
use forgellm_frontend::weight_loader::ModelWeights;

use crate::kernels;
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

    // Gemma-1 scales the embedding by sqrt(hidden_size) after lookup.
    // Doing it here (not to the stored weight) preserves tied embeddings for
    // the final logit projection.
    if matches!(
        config.architecture,
        forgellm_frontend::ir::Architecture::Gemma
    ) {
        let scale = (hidden as f32).sqrt();
        for v in hidden_state.iter_mut() {
            *v *= scale;
        }
    }

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

        // Add QKV biases if present (Qwen2 uses biases on QKV)
        if let Some(q_bias) = weights.get(&format!("{prefix}.self_attn.q_proj.bias")) {
            elementwise_add_inplace(&mut q, q_bias);
        }
        if let Some(k_bias) = weights.get(&format!("{prefix}.self_attn.k_proj.bias")) {
            elementwise_add_inplace(&mut k, k_bias);
        }
        if let Some(v_bias) = weights.get(&format!("{prefix}.self_attn.v_proj.bias")) {
            elementwise_add_inplace(&mut v, v_bias);
        }

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
        match config.hidden_activation {
            forgellm_frontend::ir::HiddenActivation::SiLU => silu(&mut gate_act, &gate),
            forgellm_frontend::ir::HiddenActivation::GeluApprox => {
                kernels::gelu(&mut gate_act, &gate)
            }
        }
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

// --- Kernel wrappers (delegate to optimized kernels module) ---

fn rms_norm(output: &mut [f32], input: &[f32], weight: &[f32], eps: f32) {
    kernels::rms_norm(output, input, weight, eps);
}

fn matmul(output: &mut [f32], input: &[f32], weight: &[f32], m: usize, k: usize, n: usize) {
    kernels::matmul(output, input, weight, m, k, n);
}

fn silu(output: &mut [f32], input: &[f32]) {
    kernels::silu(output, input);
}

fn elementwise_mul(output: &mut [f32], a: &[f32], b: &[f32]) {
    kernels::elementwise_mul(output, a, b);
}

fn elementwise_add(output: &mut [f32], a: &[f32], b: &[f32]) {
    kernels::elementwise_add(output, a, b);
}

fn elementwise_add_inplace(a: &mut [f32], b: &[f32]) {
    for i in 0..a.len() {
        a[i] += b[i];
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
    kernels::attention(
        output,
        q,
        k_cache,
        v_cache,
        params.seq_len,
        params.num_heads,
        params.num_kv_heads,
        params.head_dim,
    );
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
        kernels::softmax(&mut values);
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
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
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
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
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

    // ── Real-world validation tests ──────────────────────────────────────

    /// Build a tiny model with distinguishable per-token embeddings so that
    /// different token IDs produce different logit distributions.
    fn tiny_model_with_varied_weights() -> (ModelConfig, Graph, ModelWeights) {
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
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        };

        let graph = forgellm_frontend::graph_builder::build_graph(&config).unwrap();

        let h = 8;
        let inter = 16;
        let vocab = 16;
        let num_heads = 2;
        let num_kv_heads = 1;
        let head_dim = 4;

        let mut tensors = HashMap::new();

        // Varied embeddings: each token gets a distinct embedding vector
        let mut embed = vec![0.0f32; vocab * h];
        for tok in 0..vocab {
            for d in 0..h {
                embed[tok * h + d] = ((tok * h + d) as f32 + 1.0) * 0.05;
            }
        }
        tensors.insert("model.embed_tokens.weight".into(), embed);

        tensors.insert(
            "model.layers.0.input_layernorm.weight".into(),
            vec![1.0f32; h],
        );
        // Use varied projection weights so the model isn't degenerate
        let q_w: Vec<f32> = (0..num_heads * head_dim * h)
            .map(|i| ((i % 7) as f32 + 1.0) * 0.01)
            .collect();
        let k_w: Vec<f32> = (0..num_kv_heads * head_dim * h)
            .map(|i| ((i % 5) as f32 + 1.0) * 0.01)
            .collect();
        let v_w: Vec<f32> = (0..num_kv_heads * head_dim * h)
            .map(|i| ((i % 3) as f32 + 1.0) * 0.01)
            .collect();
        let o_w: Vec<f32> = (0..h * num_heads * head_dim)
            .map(|i| ((i % 11) as f32 + 1.0) * 0.01)
            .collect();
        tensors.insert("model.layers.0.self_attn.q_proj.weight".into(), q_w);
        tensors.insert("model.layers.0.self_attn.k_proj.weight".into(), k_w);
        tensors.insert("model.layers.0.self_attn.v_proj.weight".into(), v_w);
        tensors.insert("model.layers.0.self_attn.o_proj.weight".into(), o_w);
        tensors.insert(
            "model.layers.0.post_attention_layernorm.weight".into(),
            vec![1.0f32; h],
        );

        let gate_w: Vec<f32> = (0..inter * h)
            .map(|i| ((i % 13) as f32 + 1.0) * 0.01)
            .collect();
        let up_w: Vec<f32> = (0..inter * h)
            .map(|i| ((i % 9) as f32 + 1.0) * 0.01)
            .collect();
        let down_w: Vec<f32> = (0..h * inter)
            .map(|i| ((i % 7) as f32 + 1.0) * 0.01)
            .collect();
        tensors.insert("model.layers.0.mlp.gate_proj.weight".into(), gate_w);
        tensors.insert("model.layers.0.mlp.up_proj.weight".into(), up_w);
        tensors.insert("model.layers.0.mlp.down_proj.weight".into(), down_w);

        tensors.insert("model.norm.weight".into(), vec![1.0f32; h]);
        // Varied lm_head so different hidden states map to different logits
        let lm_head: Vec<f32> = (0..vocab * h)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
            .collect();
        tensors.insert("lm_head.weight".into(), lm_head);

        let weights = ModelWeights { tensors };
        (config, graph, weights)
    }

    #[test]
    fn different_prompts_produce_different_logits() {
        // Two different token IDs at pos=0 should produce different logit vectors.
        // This validates that the model distinguishes inputs (not degenerate).
        let (_config, graph, weights) = tiny_model_with_varied_weights();

        let mut cache1 = KVCache::new(1, 1, 4);
        let logits1 = forward(0, 0, &graph, &weights, &mut cache1);

        let mut cache2 = KVCache::new(1, 1, 4);
        let logits2 = forward(5, 0, &graph, &weights, &mut cache2);

        // Both should be finite
        for &l in &logits1 {
            assert!(l.is_finite(), "logits1 contains non-finite value: {l}");
        }
        for &l in &logits2 {
            assert!(l.is_finite(), "logits2 contains non-finite value: {l}");
        }

        // They should differ (the model is not degenerate)
        let differs = logits1
            .iter()
            .zip(logits2.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "different input tokens should produce different logit distributions"
        );
    }

    #[test]
    fn cache_reset_produces_same_logits() {
        // After clearing the cache, running the same token at pos=0 should
        // produce identical logits as a fresh run. This validates that clear()
        // truly resets all state for independent multi-request serving.
        let (_config, graph, weights) = tiny_model_with_varied_weights();

        // First run: fresh cache
        let mut cache = KVCache::new(1, 1, 4);
        let logits_fresh = forward(3, 0, &graph, &weights, &mut cache);

        // Advance the cache with more tokens to build up state
        cache.advance();
        let _ = forward(7, 1, &graph, &weights, &mut cache);
        cache.advance();
        assert_eq!(cache.len(), 2);

        // Clear and re-run
        cache.clear();
        assert_eq!(cache.len(), 0);
        let logits_after_reset = forward(3, 0, &graph, &weights, &mut cache);

        // Should be identical
        for (i, (a, b)) in logits_fresh
            .iter()
            .zip(logits_after_reset.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "logit[{i}] differs after reset: fresh={a}, after_reset={b}"
            );
        }
    }

    #[test]
    fn forward_at_pos_zero_no_nan() {
        // pos=0 is the first token where seq_len=1 in attention.
        // This is a common edge case: softmax over a single element,
        // RoPE with angle=0, and KV cache with one entry.
        let (_config, graph, weights) = tiny_model_with_varied_weights();
        let mut cache = KVCache::new(1, 1, 4);

        let logits = forward(0, 0, &graph, &weights, &mut cache);
        assert_eq!(logits.len(), 16);

        for (i, &l) in logits.iter().enumerate() {
            assert!(
                !l.is_nan(),
                "logit[{i}] is NaN at pos=0 — likely a softmax or attention bug"
            );
            assert!(!l.is_infinite(), "logit[{i}] is infinite at pos=0");
        }
    }
}
