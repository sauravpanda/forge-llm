//! Operator fusion passes.
//!
//! Scans the computation graph for patterns of operations that can be
//! merged into single fused kernels to reduce memory bandwidth.

use forgellm_frontend::ir::*;
use std::collections::{HashMap, HashSet};

/// A fused operation that replaces multiple IR nodes.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum FusedOp {
    /// Fused RMSNorm followed by MatMul.
    /// Eliminates the intermediate normalized tensor.
    RMSNormMatMul {
        eps: f32,
        /// Shape of the weight matrix [out_features, in_features].
        weight_shape: Shape,
    },

    /// Fused gate projection + SiLU activation + up projection + elementwise multiply.
    /// Replaces: MatMul(gate) -> SiLU -> MatMul(up) -> Mul
    GateUpSiLU {
        intermediate_size: usize,
        hidden_size: usize,
    },
}

/// Record of a fusion that was applied.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FusionRecord {
    /// The fused operation type.
    pub fused_op: FusedOp,
    /// Node IDs that were fused (will be replaced).
    pub fused_nodes: Vec<NodeId>,
    /// The replacement node ID in the new graph.
    pub replacement_node: NodeId,
}

/// Result of running fusion passes on a graph.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FusionResult {
    pub graph: Graph,
    pub fusions: Vec<FusionRecord>,
}

/// Find RMSNorm → MatMul patterns.
///
/// Pattern: RMSNorm(input, weight) → MatMul(normed, proj_weight)
/// The intermediate `normed` tensor is only consumed by the MatMul.
pub fn find_rmsnorm_matmul_patterns(graph: &Graph) -> Vec<(NodeId, NodeId)> {
    let consumers = build_consumer_map(graph);
    let mut patterns = Vec::new();

    for node in &graph.nodes {
        if let Op::RMSNorm { .. } = &node.op {
            // Check if this norm's output is consumed by exactly one MatMul
            if let Some(consumer_ids) = consumers.get(&node.id) {
                if consumer_ids.len() == 1 {
                    let consumer = &graph.nodes[consumer_ids[0]];
                    if matches!(consumer.op, Op::MatMul) && consumer.inputs[0] == node.id {
                        patterns.push((node.id, consumer.id));
                    }
                }
            }
        }
    }

    patterns
}

/// Find Gate+SiLU+Up+Mul patterns in FFN blocks.
///
/// Pattern:
///   gate = MatMul(input, gate_weight)
///   gate_act = SiLU(gate)
///   up = MatMul(input, up_weight)
///   result = Mul(gate_act, up)
///
/// Both MatMuls must share the same input (the normed hidden state).
pub fn find_gate_up_silu_patterns(graph: &Graph) -> Vec<(NodeId, NodeId, NodeId, NodeId)> {
    let consumers = build_consumer_map(graph);
    let mut patterns = Vec::new();

    for node in &graph.nodes {
        // Look for Mul nodes
        if !matches!(node.op, Op::Mul) || node.inputs.len() != 2 {
            continue;
        }

        let mul_id = node.id;
        let input_a = node.inputs[0];
        let input_b = node.inputs[1];

        // One input should be SiLU, the other should be MatMul
        let (silu_id, up_id) = {
            let a_op = &graph.nodes[input_a].op;
            let b_op = &graph.nodes[input_b].op;
            match (a_op, b_op) {
                (Op::SiLU, Op::MatMul) => (input_a, input_b),
                (Op::MatMul, Op::SiLU) => (input_b, input_a),
                _ => continue,
            }
        };

        // SiLU's input should be a MatMul (gate projection)
        let silu_node = &graph.nodes[silu_id];
        if silu_node.inputs.len() != 1 {
            continue;
        }
        let gate_id = silu_node.inputs[0];
        if !matches!(graph.nodes[gate_id].op, Op::MatMul) {
            continue;
        }

        // Both gate and up MatMuls should share the same first input (normed hidden state)
        let gate_node = &graph.nodes[gate_id];
        let up_node = &graph.nodes[up_id];
        if gate_node.inputs[0] != up_node.inputs[0] {
            continue;
        }

        // Verify the SiLU is only consumed by this Mul
        if let Some(silu_consumers) = consumers.get(&silu_id) {
            if silu_consumers.len() != 1 {
                continue;
            }
        }

        patterns.push((gate_id, silu_id, up_id, mul_id));
    }

    patterns
}

/// Build a map of node ID → list of nodes that consume its output.
fn build_consumer_map(graph: &Graph) -> HashMap<NodeId, Vec<NodeId>> {
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for node in &graph.nodes {
        for &input_id in &node.inputs {
            consumers.entry(input_id).or_default().push(node.id);
        }
    }
    consumers
}

/// Count how many fusion opportunities exist in a graph.
pub fn count_fusion_opportunities(graph: &Graph) -> FusionStats {
    let rmsnorm_matmul = find_rmsnorm_matmul_patterns(graph);
    let gate_up_silu = find_gate_up_silu_patterns(graph);

    FusionStats {
        rmsnorm_matmul_count: rmsnorm_matmul.len(),
        gate_up_silu_count: gate_up_silu.len(),
        total: rmsnorm_matmul.len() + gate_up_silu.len(),
    }
}

/// Statistics about fusion opportunities in a graph.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionStats {
    pub rmsnorm_matmul_count: usize,
    pub gate_up_silu_count: usize,
    pub total: usize,
}

/// Eliminate dead nodes (nodes whose outputs are never consumed).
/// Returns the set of live node IDs.
pub fn find_live_nodes(graph: &Graph) -> HashSet<NodeId> {
    if graph.is_empty() {
        return HashSet::new();
    }

    let mut live = HashSet::new();
    // The last node (output) is always live
    let output_id = graph.nodes.len() - 1;
    live.insert(output_id);

    // Walk backwards marking all inputs as live
    for node in graph.nodes.iter().rev() {
        if live.contains(&node.id) {
            for &input_id in &node.inputs {
                live.insert(input_id);
            }
        }
    }

    live
}

#[cfg(test)]
mod tests {
    use super::*;
    use forgellm_frontend::graph_builder;

    fn llama_tiny_config() -> ModelConfig {
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
    fn rmsnorm_matmul_not_found_for_multi_consumer() {
        let config = llama_tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let patterns = find_rmsnorm_matmul_patterns(&graph);

        // In Llama, attention norm feeds Q/K/V (3 consumers) and
        // FFN norm feeds gate/up (2 consumers), so no single-consumer
        // RMSNorm→MatMul patterns exist. The final norm feeds LogitsProjection,
        // not MatMul, so it doesn't match either.
        assert_eq!(
            patterns.len(),
            0,
            "no single-consumer RMSNorm+MatMul in standard Llama"
        );
    }

    #[test]
    fn find_gate_up_silu_in_llama() {
        let config = llama_tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let patterns = find_gate_up_silu_patterns(&graph);

        // Each layer has one gate+SiLU+up+mul pattern in the FFN
        assert_eq!(
            patterns.len(),
            config.num_layers,
            "should find one GateUpSiLU pattern per layer"
        );
    }

    #[test]
    fn fusion_stats() {
        let config = llama_tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let stats = count_fusion_opportunities(&graph);

        assert!(stats.total > 0);
        assert_eq!(stats.gate_up_silu_count, config.num_layers);
    }

    #[test]
    fn all_nodes_are_live() {
        let config = llama_tiny_config();
        let graph = graph_builder::build_graph(&config).unwrap();
        let live = find_live_nodes(&graph);

        // In a well-constructed graph, all nodes should be reachable from the output
        assert_eq!(
            live.len(),
            graph.len(),
            "all nodes should be live in a valid model graph"
        );
    }

    #[test]
    fn consumer_map_is_correct() {
        let mut graph = Graph::new("test");
        let a = graph.input("a", Shape::new(vec![4]), DType::F32);
        let b = graph.input("b", Shape::new(vec![4]), DType::F32);
        let tid = graph.alloc_tensor_id();
        let c = graph.add_node(
            Op::Add,
            vec![a, b],
            TensorInfo {
                id: tid,
                name: "c".into(),
                shape: Shape::new(vec![4]),
                dtype: DType::F32,
            },
        );

        let consumers = build_consumer_map(&graph);
        assert_eq!(consumers.get(&a), Some(&vec![c]));
        assert_eq!(consumers.get(&b), Some(&vec![c]));
        assert_eq!(consumers.get(&c), None); // c has no consumers
    }

    #[test]
    fn dead_node_detection() {
        let mut graph = Graph::new("test");
        let a = graph.input("a", Shape::new(vec![4]), DType::F32);
        let _b = graph.input("b", Shape::new(vec![4]), DType::F32); // dead node

        let tid = graph.alloc_tensor_id();
        // Only uses 'a', so 'b' is dead
        let _c = graph.add_node(
            Op::SiLU,
            vec![a],
            TensorInfo {
                id: tid,
                name: "c".into(),
                shape: Shape::new(vec![4]),
                dtype: DType::F32,
            },
        );

        let live = find_live_nodes(&graph);
        assert!(live.contains(&a)); // a is used by c
        assert!(!live.contains(&_b)); // b is dead
        assert!(live.contains(&_c)); // c is the output
    }
}
