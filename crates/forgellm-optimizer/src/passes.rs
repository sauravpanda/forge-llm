//! Optimization pass orchestration.
//!
//! Runs all optimization passes in sequence on a computation graph.

use forgellm_frontend::ir::Graph;
use tracing::info;

use crate::fusion;

/// Optimization configuration.
#[derive(Debug, Clone)]
pub struct OptimizeConfig {
    /// Enable operator fusion passes.
    pub enable_fusion: bool,
    /// Enable dead node elimination.
    pub enable_dce: bool,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            enable_fusion: true,
            enable_dce: true,
        }
    }
}

/// Apply all optimization passes to a computation graph.
///
/// Currently performs analysis only (identifies fusion opportunities).
/// Future versions will rewrite the graph with fused operations.
pub fn optimize(graph: &Graph) -> Graph {
    optimize_with_config(graph, &OptimizeConfig::default())
}

/// Apply optimization passes with explicit configuration.
pub fn optimize_with_config(graph: &Graph, config: &OptimizeConfig) -> Graph {
    let result = graph.clone();

    if config.enable_fusion {
        let stats = fusion::count_fusion_opportunities(&result);
        info!(
            rmsnorm_matmul = stats.rmsnorm_matmul_count,
            gate_up_silu = stats.gate_up_silu_count,
            total = stats.total,
            "fusion opportunities identified"
        );
    }

    if config.enable_dce {
        let live = fusion::find_live_nodes(&result);
        let dead_count = result.len() - live.len();
        if dead_count > 0 {
            info!(dead_nodes = dead_count, "dead nodes found");
        }
    }

    // For now, return the graph unchanged.
    // Graph rewriting with fused ops will be added incrementally.
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use forgellm_frontend::graph_builder;
    use forgellm_frontend::ir::*;

    #[test]
    fn optimize_preserves_graph() {
        let config = ModelConfig {
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
        };

        let graph = graph_builder::build_graph(&config).unwrap();
        let optimized = optimize(&graph);

        assert_eq!(optimized.len(), graph.len());
        assert_eq!(optimized.name, graph.name);
        assert!(optimized.validate().is_ok());
    }

    #[test]
    fn optimize_with_fusion_disabled() {
        let config = ModelConfig {
            architecture: Architecture::Llama,
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
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        };

        let graph = graph_builder::build_graph(&config).unwrap();
        let opt_config = OptimizeConfig {
            enable_fusion: false,
            enable_dce: false,
        };
        let optimized = optimize_with_config(&graph, &opt_config);
        assert_eq!(optimized.len(), graph.len());
    }

    #[test]
    fn optimize_empty_graph() {
        let graph = Graph::new("empty");
        let optimized = optimize(&graph);
        assert!(optimized.is_empty());
    }
}
