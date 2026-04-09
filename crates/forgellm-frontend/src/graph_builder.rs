//! Graph builder — constructs IR computation graphs from model configs.
//!
//! Takes a `ModelConfig` (extracted from GGUF metadata or HF config.json)
//! and builds the full transformer computation graph with all layers,
//! weight references, and operations.

use crate::ir::*;

/// Build a complete computation graph for a transformer model.
///
/// The graph represents the full forward pass from token IDs to logits.
/// Weight names follow the HuggingFace convention (model.layers.N.*).
pub fn build_graph(config: &ModelConfig) -> Result<Graph, GraphBuildError> {
    match config.architecture {
        Architecture::Llama | Architecture::Mistral => build_llama_graph(config),
        Architecture::Qwen2 => build_llama_graph(config), // Qwen2 is structurally identical to Llama
        ref arch => Err(GraphBuildError::UnsupportedArchitecture(arch.to_string())),
    }
}

/// Build a Llama-family computation graph.
///
/// Architecture: embedding → N × (attention_norm → attention → residual →
/// ffn_norm → ffn → residual) → final_norm → lm_head
fn build_llama_graph(config: &ModelConfig) -> Result<Graph, GraphBuildError> {
    let mut graph =
        Graph::new(format!("{}-graph", config.architecture)).with_config(config.clone());

    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let dtype = config.dtype;

    // Input: token IDs [batch, seq_len]
    let input_ids = graph.input("input_ids", Shape::new(vec![1, 0]), DType::I32);

    // Token embedding: [vocab_size, hidden_size]
    let embed_weight = graph.load_weight(
        "model.embed_tokens.weight",
        Shape::new(vec![vocab, hidden]),
        dtype,
    );

    let tid = graph.alloc_tensor_id();
    let mut current = graph.add_node(
        Op::Embedding {
            vocab_size: vocab,
            embed_dim: hidden,
        },
        vec![input_ids, embed_weight],
        TensorInfo {
            id: tid,
            name: "embed_output".into(),
            shape: Shape::new(vec![1, 0, hidden]),
            dtype,
        },
    );

    // Transformer layers
    for layer_idx in 0..config.num_layers {
        let prefix = format!("model.layers.{layer_idx}");
        current = build_llama_layer(&mut graph, config, &prefix, current)?;
    }

    // Final RMSNorm
    let final_norm_w = graph.load_weight("model.norm.weight", Shape::new(vec![hidden]), dtype);
    let tid = graph.alloc_tensor_id();
    let normed = graph.add_node(
        Op::RMSNorm {
            eps: config.rms_norm_eps,
        },
        vec![current, final_norm_w],
        TensorInfo {
            id: tid,
            name: "final_norm".into(),
            shape: Shape::new(vec![1, 0, hidden]),
            dtype,
        },
    );

    // LM head (logits projection)
    let lm_head_weight =
        graph.load_weight("lm_head.weight", Shape::new(vec![vocab, hidden]), dtype);
    let tid = graph.alloc_tensor_id();
    let _logits = graph.add_node(
        Op::LogitsProjection { vocab_size: vocab },
        vec![normed, lm_head_weight],
        TensorInfo {
            id: tid,
            name: "logits".into(),
            shape: Shape::new(vec![1, 0, vocab]),
            dtype: DType::F32, // Logits are always f32
        },
    );

    graph.validate().map_err(GraphBuildError::Validation)?;
    Ok(graph)
}

/// Build a single Llama transformer layer.
///
/// Structure: input_norm → attention → residual → ffn_norm → FFN → residual
fn build_llama_layer(
    graph: &mut Graph,
    config: &ModelConfig,
    prefix: &str,
    input: NodeId,
) -> Result<NodeId, GraphBuildError> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let dtype = config.dtype;

    // === Self-Attention ===

    // Input LayerNorm
    let attn_norm_w = graph.load_weight(
        format!("{prefix}.input_layernorm.weight"),
        Shape::new(vec![hidden]),
        dtype,
    );
    let tid = graph.alloc_tensor_id();
    let normed = graph.add_node(
        Op::RMSNorm {
            eps: config.rms_norm_eps,
        },
        vec![input, attn_norm_w],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.attn_norm"),
            shape: Shape::new(vec![1, 0, hidden]),
            dtype,
        },
    );

    // Q, K, V projections
    let q_weight = graph.load_weight(
        format!("{prefix}.self_attn.q_proj.weight"),
        Shape::new(vec![num_heads * head_dim, hidden]),
        dtype,
    );
    let tid = graph.alloc_tensor_id();
    let q = graph.add_node(
        Op::MatMul,
        vec![normed, q_weight],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.q_proj"),
            shape: Shape::new(vec![1, 0, num_heads * head_dim]),
            dtype,
        },
    );

    let k_weight = graph.load_weight(
        format!("{prefix}.self_attn.k_proj.weight"),
        Shape::new(vec![num_kv_heads * head_dim, hidden]),
        dtype,
    );
    let tid = graph.alloc_tensor_id();
    let k = graph.add_node(
        Op::MatMul,
        vec![normed, k_weight],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.k_proj"),
            shape: Shape::new(vec![1, 0, num_kv_heads * head_dim]),
            dtype,
        },
    );

    let v_weight = graph.load_weight(
        format!("{prefix}.self_attn.v_proj.weight"),
        Shape::new(vec![num_kv_heads * head_dim, hidden]),
        dtype,
    );
    let tid = graph.alloc_tensor_id();
    let v = graph.add_node(
        Op::MatMul,
        vec![normed, v_weight],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.v_proj"),
            shape: Shape::new(vec![1, 0, num_kv_heads * head_dim]),
            dtype,
        },
    );

    // RoPE on Q and K
    let tid = graph.alloc_tensor_id();
    let q_rope = graph.add_node(
        Op::RoPE {
            max_seq_len: config.max_seq_len,
            rope_theta: config.rope_theta,
            head_dim,
        },
        vec![q],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.q_rope"),
            shape: Shape::new(vec![1, 0, num_heads * head_dim]),
            dtype,
        },
    );

    let tid = graph.alloc_tensor_id();
    let k_rope = graph.add_node(
        Op::RoPE {
            max_seq_len: config.max_seq_len,
            rope_theta: config.rope_theta,
            head_dim,
        },
        vec![k],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.k_rope"),
            shape: Shape::new(vec![1, 0, num_kv_heads * head_dim]),
            dtype,
        },
    );

    // Attention
    let tid = graph.alloc_tensor_id();
    let attn_out = graph.add_node(
        Op::Attention {
            num_heads,
            num_kv_heads,
            head_dim,
        },
        vec![q_rope, k_rope, v],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.attn"),
            shape: Shape::new(vec![1, 0, num_heads * head_dim]),
            dtype,
        },
    );

    // Output projection
    let o_weight = graph.load_weight(
        format!("{prefix}.self_attn.o_proj.weight"),
        Shape::new(vec![hidden, num_heads * head_dim]),
        dtype,
    );
    let tid = graph.alloc_tensor_id();
    let attn_proj = graph.add_node(
        Op::MatMul,
        vec![attn_out, o_weight],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.o_proj"),
            shape: Shape::new(vec![1, 0, hidden]),
            dtype,
        },
    );

    // Residual connection
    let tid = graph.alloc_tensor_id();
    let after_attn = graph.add_node(
        Op::Residual,
        vec![input, attn_proj],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.attn_residual"),
            shape: Shape::new(vec![1, 0, hidden]),
            dtype,
        },
    );

    // === Feed-Forward Network ===

    // Post-attention LayerNorm
    let ffn_norm_w = graph.load_weight(
        format!("{prefix}.post_attention_layernorm.weight"),
        Shape::new(vec![hidden]),
        dtype,
    );
    let tid = graph.alloc_tensor_id();
    let ffn_normed = graph.add_node(
        Op::RMSNorm {
            eps: config.rms_norm_eps,
        },
        vec![after_attn, ffn_norm_w],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.ffn_norm"),
            shape: Shape::new(vec![1, 0, hidden]),
            dtype,
        },
    );

    // Gate projection (SiLU-gated FFN)
    let gate_weight = graph.load_weight(
        format!("{prefix}.mlp.gate_proj.weight"),
        Shape::new(vec![intermediate, hidden]),
        dtype,
    );
    let tid = graph.alloc_tensor_id();
    let gate = graph.add_node(
        Op::MatMul,
        vec![ffn_normed, gate_weight],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.gate_proj"),
            shape: Shape::new(vec![1, 0, intermediate]),
            dtype,
        },
    );

    // SiLU activation on gate
    let tid = graph.alloc_tensor_id();
    let gate_act = graph.add_node(
        Op::SiLU,
        vec![gate],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.gate_silu"),
            shape: Shape::new(vec![1, 0, intermediate]),
            dtype,
        },
    );

    // Up projection
    let up_weight = graph.load_weight(
        format!("{prefix}.mlp.up_proj.weight"),
        Shape::new(vec![intermediate, hidden]),
        dtype,
    );
    let tid = graph.alloc_tensor_id();
    let up = graph.add_node(
        Op::MatMul,
        vec![ffn_normed, up_weight],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.up_proj"),
            shape: Shape::new(vec![1, 0, intermediate]),
            dtype,
        },
    );

    // Gate * Up (elementwise multiply)
    let tid = graph.alloc_tensor_id();
    let ffn_hidden = graph.add_node(
        Op::Mul,
        vec![gate_act, up],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.gate_up_mul"),
            shape: Shape::new(vec![1, 0, intermediate]),
            dtype,
        },
    );

    // Down projection
    let down_weight = graph.load_weight(
        format!("{prefix}.mlp.down_proj.weight"),
        Shape::new(vec![hidden, intermediate]),
        dtype,
    );
    let tid = graph.alloc_tensor_id();
    let ffn_out = graph.add_node(
        Op::MatMul,
        vec![ffn_hidden, down_weight],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.down_proj"),
            shape: Shape::new(vec![1, 0, hidden]),
            dtype,
        },
    );

    // Residual connection
    let tid = graph.alloc_tensor_id();
    let output = graph.add_node(
        Op::Residual,
        vec![after_attn, ffn_out],
        TensorInfo {
            id: tid,
            name: format!("{prefix}.ffn_residual"),
            shape: Shape::new(vec![1, 0, hidden]),
            dtype,
        },
    );

    Ok(output)
}

/// Errors in graph construction.
#[derive(Debug, thiserror::Error)]
pub enum GraphBuildError {
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("graph validation failed: {0}")]
    Validation(#[from] GraphError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_1b_config() -> ModelConfig {
        ModelConfig {
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
        }
    }

    fn smollm_135m_config() -> ModelConfig {
        ModelConfig {
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
        }
    }

    #[test]
    fn build_llama_1b_graph() {
        let config = llama_1b_config();
        let graph = build_graph(&config).unwrap();

        assert!(!graph.is_empty());
        assert!(graph.config.is_some());
        assert!(graph.validate().is_ok());

        // Check we have weights registered
        assert!(graph.weights.contains_key("model.embed_tokens.weight"));
        assert!(graph.weights.contains_key("model.norm.weight"));
        assert!(graph.weights.contains_key("lm_head.weight"));
        assert!(graph
            .weights
            .contains_key("model.layers.0.input_layernorm.weight"));
        assert!(graph
            .weights
            .contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(graph
            .weights
            .contains_key("model.layers.0.mlp.gate_proj.weight"));
        assert!(graph
            .weights
            .contains_key("model.layers.15.mlp.down_proj.weight"));
    }

    #[test]
    fn build_smollm_135m_graph() {
        let config = smollm_135m_config();
        let graph = build_graph(&config).unwrap();

        assert!(graph.validate().is_ok());
        assert!(graph
            .weights
            .contains_key("model.layers.29.mlp.down_proj.weight"));

        // Verify weight shapes for sub-1B model
        let embed = &graph.weights["model.embed_tokens.weight"];
        assert_eq!(embed.shape, Shape::new(vec![49152, 576]));

        let q_proj = &graph.weights["model.layers.0.self_attn.q_proj.weight"];
        assert_eq!(q_proj.shape, Shape::new(vec![576, 576])); // 9 heads * 64 head_dim = 576
    }

    #[test]
    fn graph_node_count() {
        let config = llama_1b_config();
        let graph = build_graph(&config).unwrap();

        // Per layer: 2 norms + 4 QKV projections + 2 RoPE + 1 attention +
        //            1 output proj + 1 residual + 3 FFN projections +
        //            1 SiLU + 1 mul + 1 residual + 6 weight loads + 2 norm weights = ~23
        // Plus: 1 input + 1 embed_weight + 1 embedding + 1 final_norm_weight +
        //       1 final_norm + 1 lm_head_weight + 1 lm_head = 7
        // Total should be reasonable for 16 layers
        assert!(graph.len() > 100);
    }

    #[test]
    fn graph_has_correct_output() {
        let config = llama_1b_config();
        let graph = build_graph(&config).unwrap();

        // Last node should be logits projection
        let last = graph.node(graph.len() - 1);
        assert!(matches!(last.op, Op::LogitsProjection { .. }));
        assert_eq!(last.output.dtype, DType::F32);
    }

    #[test]
    fn qwen2_uses_llama_builder() {
        let config = ModelConfig {
            architecture: Architecture::Qwen2,
            hidden_size: 1536,
            intermediate_size: 8960,
            num_layers: 28,
            num_attention_heads: 12,
            num_kv_heads: 2,
            head_dim: 128,
            vocab_size: 151936,
            max_seq_len: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            dtype: DType::BF16,
        };

        let graph = build_graph(&config).unwrap();
        assert!(graph.validate().is_ok());
        assert!(graph
            .weights
            .contains_key("model.layers.27.mlp.down_proj.weight"));
    }

    #[test]
    fn unsupported_architecture_errors() {
        let config = ModelConfig {
            architecture: Architecture::Phi3,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,
            head_dim: 96,
            vocab_size: 32064,
            max_seq_len: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::F16,
        };

        let result = build_graph(&config);
        assert!(matches!(
            result,
            Err(GraphBuildError::UnsupportedArchitecture(_))
        ));
    }

    #[test]
    fn topological_order_is_valid() {
        let config = smollm_135m_config();
        let graph = build_graph(&config).unwrap();

        // Every node's inputs should have lower IDs (already enforced by validate)
        for node in &graph.nodes {
            for &input_id in &node.inputs {
                assert!(
                    input_id < node.id,
                    "node {} references future node {}",
                    node.id,
                    input_id
                );
            }
        }
    }
}
