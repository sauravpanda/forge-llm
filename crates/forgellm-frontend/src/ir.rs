//! Intermediate representation for transformer computation graphs.
//!
//! The IR is the central abstraction in ForgeLLM: all frontends produce it,
//! all backends consume it. It represents a static computation graph with
//! typed tensor operations specialized for transformer architectures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Unique identifier for a node in the computation graph.
pub type NodeId = usize;

/// Unique identifier for a tensor.
pub type TensorId = usize;

/// Scalar data type for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// 8-bit float (E4M3)
    F8E4M3,
    /// 8-bit float (E5M2)
    F8E5M2,
    /// 8-bit quantized (block-wise, with scale factors)
    Q8_0,
    /// 4-bit quantized (block-wise, with scale factors)
    Q4_0,
    /// 4-bit quantized (with scale + min)
    Q4_1,
    /// 4-bit K-quant — 256-element super-block, per-sub-block 6-bit scale + 6-bit min.
    Q4_K,
    /// 6-bit K-quant — 256-element super-block, per-16-elem int8 scale, no min.
    Q6_K,
    /// 2-bit quantized
    Q2,
    /// 4-bit NormalFloat (for QLoRA)
    NF4,
    I32,
    I64,
}

impl DType {
    /// Size in bytes for non-quantized types, or effective bits per element for quantized.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F8E4M3 | DType::F8E5M2 | DType::Q8_0 => 1,
            DType::I64 => 8,
            // Quantized types: return 1 as a placeholder; actual size depends on block size
            DType::Q4_0 | DType::Q4_1 | DType::Q4_K | DType::NF4 => 1,
            DType::Q6_K => 1,
            DType::Q2 => 1,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            DType::Q8_0 | DType::Q4_0 | DType::Q4_1 | DType::Q4_K | DType::Q2 | DType::NF4
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DType::F32 | DType::F16 | DType::BF16 | DType::F8E4M3 | DType::F8E5M2
        )
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::F8E4M3 => write!(f, "f8e4m3"),
            DType::F8E5M2 => write!(f, "f8e5m2"),
            DType::Q8_0 => write!(f, "q8_0"),
            DType::Q4_0 => write!(f, "q4_0"),
            DType::Q4_1 => write!(f, "q4_1"),
            DType::Q4_K => write!(f, "q4_k"),
            DType::Q6_K => write!(f, "q6_k"),
            DType::Q2 => write!(f, "q2"),
            DType::NF4 => write!(f, "nf4"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
        }
    }
}

/// Shape of a tensor — a list of dimension sizes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims)
    }

    pub fn scalar() -> Self {
        Self(vec![])
    }

    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Returns the size of a specific dimension.
    pub fn dim(&self, i: usize) -> usize {
        self.0[i]
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// Metadata about a tensor (shape + dtype), without the actual data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInfo {
    pub id: TensorId,
    pub name: String,
    pub shape: Shape,
    pub dtype: DType,
}

/// An operation in the computation graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    // --- Tensor creation ---
    /// Load a constant weight tensor (by name/id).
    LoadWeight {
        name: String,
    },

    /// External input (e.g., token ids).
    Input {
        name: String,
    },

    // --- Core linear algebra ---
    /// Matrix multiplication: (M, K) x (K, N) -> (M, N)
    MatMul,

    /// Batched matrix multiplication.
    BatchMatMul,

    // --- Elementwise operations ---
    Add,
    Mul,
    /// Sigmoid Linear Unit: x * sigmoid(x)
    SiLU,
    /// Gaussian Error Linear Unit
    GeLU,
    /// Rectified Linear Unit
    ReLU,

    // --- Normalization ---
    /// Root Mean Square Layer Normalization with epsilon.
    RMSNorm {
        eps: f32,
    },

    /// Layer Normalization with epsilon.
    LayerNorm {
        eps: f32,
    },

    // --- Attention-specific ---
    /// Rotary Position Embedding.
    RoPE {
        /// Maximum sequence length.
        max_seq_len: usize,
        /// Base frequency (typically 10000.0 or 500000.0).
        rope_theta: f32,
        /// Head dimension.
        head_dim: usize,
    },

    /// Scaled dot-product attention.
    /// Inputs: Q, K, V, optional mask.
    Attention {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },

    /// Softmax along the last dimension.
    Softmax,

    // --- Shape operations ---
    /// Reshape tensor to new shape.
    Reshape {
        shape: Shape,
    },

    /// Transpose dimensions.
    Transpose {
        dim0: usize,
        dim1: usize,
    },

    /// Contiguous memory layout.
    Contiguous,

    // --- Embedding ---
    /// Token embedding lookup.
    Embedding {
        vocab_size: usize,
        embed_dim: usize,
    },

    // --- Output ---
    /// Final logits projection (often tied to embedding weights).
    LogitsProjection {
        vocab_size: usize,
    },

    // --- Residual ---
    /// Residual connection (just an Add, but semantically distinct for fusion).
    Residual,

    // --- Cast ---
    /// Cast tensor to a different dtype.
    Cast {
        to: DType,
    },
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::LoadWeight { name } => write!(f, "LoadWeight({name})"),
            Op::Input { name } => write!(f, "Input({name})"),
            Op::MatMul => write!(f, "MatMul"),
            Op::BatchMatMul => write!(f, "BatchMatMul"),
            Op::Add => write!(f, "Add"),
            Op::Mul => write!(f, "Mul"),
            Op::SiLU => write!(f, "SiLU"),
            Op::GeLU => write!(f, "GeLU"),
            Op::ReLU => write!(f, "ReLU"),
            Op::RMSNorm { eps } => write!(f, "RMSNorm(eps={eps})"),
            Op::LayerNorm { eps } => write!(f, "LayerNorm(eps={eps})"),
            Op::RoPE { head_dim, .. } => write!(f, "RoPE(head_dim={head_dim})"),
            Op::Attention {
                num_heads,
                num_kv_heads,
                head_dim,
            } => write!(f, "Attention(h={num_heads},kv={num_kv_heads},d={head_dim})"),
            Op::Softmax => write!(f, "Softmax"),
            Op::Reshape { shape } => write!(f, "Reshape({shape})"),
            Op::Transpose { dim0, dim1 } => write!(f, "Transpose({dim0},{dim1})"),
            Op::Contiguous => write!(f, "Contiguous"),
            Op::Embedding {
                vocab_size,
                embed_dim,
            } => write!(f, "Embedding(v={vocab_size},d={embed_dim})"),
            Op::LogitsProjection { vocab_size } => {
                write!(f, "LogitsProjection(v={vocab_size})")
            }
            Op::Residual => write!(f, "Residual"),
            Op::Cast { to } => write!(f, "Cast({to})"),
        }
    }
}

/// A node in the computation graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    /// The operation this node performs.
    pub op: Op,
    /// Input node IDs (ordered).
    pub inputs: Vec<NodeId>,
    /// Output tensor info.
    pub output: TensorInfo,
}

/// Model architecture type, used to select the right graph-building strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Architecture {
    Llama,
    Qwen2,
    Mistral,
    Phi3,
    Gemma,
    StableLM,
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Architecture::Llama => write!(f, "Llama"),
            Architecture::Qwen2 => write!(f, "Qwen2"),
            Architecture::Mistral => write!(f, "Mistral"),
            Architecture::Phi3 => write!(f, "Phi3"),
            Architecture::Gemma => write!(f, "Gemma"),
            Architecture::StableLM => write!(f, "StableLM"),
        }
    }
}

/// Activation used in the FFN gated-linear-unit (gate ⊙ activation(up)).
///
/// Llama/Qwen2/Mistral/Phi-3/StableLM use `SiLU` (`x * sigmoid(x)`).
/// Gemma-1 uses `GeluApprox` (tanh-based GeLU:
/// `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HiddenActivation {
    #[default]
    SiLU,
    GeluApprox,
}

/// Categories of weight tensors used to look up per-projection storage dtype.
///
/// Real GGUF Q4_K_M files mix dtypes per projection: most matmul weights are
/// Q4_K, but `attn_v` and `ffn_down` are upgraded to Q6_K for accuracy, and
/// `output.weight` is also typically Q6_K.  This enum names the buckets the
/// codegen dispatches on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjCategory {
    Embed,
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
    LmHead,
}

impl ProjCategory {
    /// Map an HF-conventionalized tensor name (the keys produced by
    /// `weight_loader::gguf_name_to_hf`, also used directly by the
    /// SafeTensors loader) to a projection category, if one applies.
    /// Returns `None` for norms / non-projection weights.
    pub fn from_hf_name(name: &str) -> Option<Self> {
        if name == "model.embed_tokens.weight" {
            return Some(Self::Embed);
        }
        if name == "lm_head.weight" || name == "output.weight" {
            return Some(Self::LmHead);
        }
        // model.layers.N.<suffix>
        let suffix = name.strip_prefix("model.layers.")?;
        let dot_pos = suffix.find('.')?;
        let suffix = &suffix[dot_pos + 1..];
        match suffix {
            "self_attn.q_proj.weight" => Some(Self::Q),
            "self_attn.k_proj.weight" => Some(Self::K),
            "self_attn.v_proj.weight" => Some(Self::V),
            "self_attn.o_proj.weight" => Some(Self::O),
            "mlp.gate_proj.weight" => Some(Self::Gate),
            "mlp.up_proj.weight" => Some(Self::Up),
            "mlp.down_proj.weight" => Some(Self::Down),
            _ => None,
        }
    }

    /// Map a raw GGUF tensor name (`token_embd.weight`, `blk.N.attn_q.weight`,
    /// `output.weight`, ...) to a projection category, if one applies.
    pub fn from_gguf_name(name: &str) -> Option<Self> {
        match name {
            "token_embd.weight" => return Some(Self::Embed),
            "output.weight" | "lm_head.weight" => return Some(Self::LmHead),
            _ => {}
        }
        let rest = name.strip_prefix("blk.")?;
        let dot_pos = rest.find('.')?;
        let suffix = &rest[dot_pos + 1..];
        match suffix {
            "attn_q.weight" => Some(Self::Q),
            "attn_k.weight" => Some(Self::K),
            "attn_v.weight" => Some(Self::V),
            "attn_output.weight" => Some(Self::O),
            "ffn_gate.weight" => Some(Self::Gate),
            "ffn_up.weight" => Some(Self::Up),
            "ffn_down.weight" => Some(Self::Down),
            _ => None,
        }
    }
}

/// Per-projection storage dtypes.
///
/// When `ModelConfig::proj_dtypes` is `Some`, each projection's matmul
/// dispatches to the kernel matching its category here.  When `None`, all
/// projections fall back to `ModelConfig::dtype` (with `lm_head_dtype` as the
/// only override) — the legacy uniform-dtype path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionDTypes {
    pub embed: DType,
    pub q: DType,
    pub k: DType,
    pub v: DType,
    pub o: DType,
    pub gate: DType,
    pub up: DType,
    pub down: DType,
    pub lm_head: DType,
}

impl ProjectionDTypes {
    /// Build a uniform projection-dtype set from a single dtype (and optional
    /// lm_head override).  This is the shape `ModelConfig::effective_proj_dtypes`
    /// returns when `proj_dtypes` is `None`.
    pub fn uniform(dtype: DType, lm_head_dtype: Option<DType>) -> Self {
        Self {
            embed: dtype,
            q: dtype,
            k: dtype,
            v: dtype,
            o: dtype,
            gate: dtype,
            up: dtype,
            down: dtype,
            lm_head: lm_head_dtype.unwrap_or(dtype),
        }
    }

    pub fn get(&self, category: ProjCategory) -> DType {
        match category {
            ProjCategory::Embed => self.embed,
            ProjCategory::Q => self.q,
            ProjCategory::K => self.k,
            ProjCategory::V => self.v,
            ProjCategory::O => self.o,
            ProjCategory::Gate => self.gate,
            ProjCategory::Up => self.up,
            ProjCategory::Down => self.down,
            ProjCategory::LmHead => self.lm_head,
        }
    }

    /// True if any projection category stores its weights as `target`.
    pub fn uses(&self, target: DType) -> bool {
        self.embed == target
            || self.q == target
            || self.k == target
            || self.v == target
            || self.o == target
            || self.gate == target
            || self.up == target
            || self.down == target
            || self.lm_head == target
    }

    /// True if all categories use the same dtype.
    pub fn is_uniform(&self) -> bool {
        let d = self.q;
        self.embed == d
            && self.k == d
            && self.v == d
            && self.o == d
            && self.gate == d
            && self.up == d
            && self.down == d
            && self.lm_head == d
    }
}

/// Configuration describing a transformer model's hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: Architecture,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub dtype: DType,
    /// Storage dtype for the lm_head / output projection.  `None` means it
    /// matches `dtype`.  GGUF Q4_K_M models typically store projection weights
    /// at a lower precision than `output.weight`, so the codegen dispatches a
    /// higher-precision kernel for the logits matmul.
    #[serde(default)]
    pub lm_head_dtype: Option<DType>,
    /// Per-projection storage dtypes.  When `Some`, each matmul dispatches
    /// to the kernel matching its category (Q4_K_M → Q4_K for most, Q6_K for
    /// `attn_v` / `ffn_down` / `output`).  When `None`, all projections share
    /// `dtype` (with `lm_head_dtype` as the only override).
    #[serde(default)]
    pub proj_dtypes: Option<ProjectionDTypes>,
    /// Sliding window attention size. `None` means full attention (Llama).
    /// `Some(n)` means each token only attends to the last `n` tokens (Mistral SWA).
    #[serde(default)]
    pub sliding_window_size: Option<usize>,
    /// Whether Q, K, V projections have bias terms. `true` for Qwen2.
    #[serde(default)]
    pub qkv_bias: bool,
    /// FFN activation (SiLU for Llama/Qwen/etc.; GeluApprox for Gemma-1).
    #[serde(default)]
    pub hidden_activation: HiddenActivation,
}

impl ModelConfig {
    /// Return the per-projection dtype set for this config.  When
    /// `proj_dtypes` is `Some`, returns it directly; otherwise builds a
    /// uniform `ProjectionDTypes` from `dtype` (+ `lm_head_dtype`).
    pub fn effective_proj_dtypes(&self) -> ProjectionDTypes {
        self.proj_dtypes
            .unwrap_or_else(|| ProjectionDTypes::uniform(self.dtype, self.lm_head_dtype))
    }

    /// Storage dtype for a single projection category.
    pub fn effective_dtype(&self, category: ProjCategory) -> DType {
        self.effective_proj_dtypes().get(category)
    }

    /// Validate that model dimensions are consistent.
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err("hidden_size must be > 0".into());
        }
        if self.num_attention_heads == 0 {
            return Err("num_attention_heads must be > 0".into());
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if self.num_kv_heads == 0 {
            return Err("num_kv_heads must be > 0".into());
        }
        if !self.num_attention_heads.is_multiple_of(self.num_kv_heads) {
            return Err(format!(
                "num_attention_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_attention_heads, self.num_kv_heads
            ));
        }
        Ok(())
    }
}

/// The computation graph — the central IR artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub name: String,
    pub config: Option<ModelConfig>,
    pub nodes: Vec<Node>,
    /// Maps weight names to their tensor info.
    pub weights: HashMap<String, TensorInfo>,
    next_node_id: NodeId,
    next_tensor_id: TensorId,
}

impl Graph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            config: None,
            nodes: Vec::new(),
            weights: HashMap::new(),
            next_node_id: 0,
            next_tensor_id: 0,
        }
    }

    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Add a node to the graph and return its ID.
    pub fn add_node(&mut self, op: Op, inputs: Vec<NodeId>, output: TensorInfo) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.push(Node {
            id,
            op,
            inputs,
            output,
        });
        id
    }

    /// Allocate a new tensor ID.
    pub fn alloc_tensor_id(&mut self) -> TensorId {
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;
        id
    }

    /// Register a weight tensor in the graph.
    pub fn register_weight(&mut self, name: String, shape: Shape, dtype: DType) -> TensorId {
        let id = self.alloc_tensor_id();
        let info = TensorInfo {
            id,
            name: name.clone(),
            shape,
            dtype,
        };
        self.weights.insert(name, info);
        id
    }

    /// Add a weight-loading node.
    pub fn load_weight(&mut self, name: impl Into<String>, shape: Shape, dtype: DType) -> NodeId {
        let name = name.into();
        let tensor_id = self.alloc_tensor_id();
        let output = TensorInfo {
            id: tensor_id,
            name: name.clone(),
            shape,
            dtype,
        };
        self.register_weight(name.clone(), output.shape.clone(), output.dtype);
        self.add_node(Op::LoadWeight { name }, vec![], output)
    }

    /// Add an input node.
    pub fn input(&mut self, name: impl Into<String>, shape: Shape, dtype: DType) -> NodeId {
        let name = name.into();
        let tensor_id = self.alloc_tensor_id();
        let output = TensorInfo {
            id: tensor_id,
            name: name.clone(),
            shape,
            dtype,
        };
        self.add_node(Op::Input { name }, vec![], output)
    }

    /// Get a node by ID.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    /// Get the output tensor info for a node.
    pub fn output_info(&self, id: NodeId) -> &TensorInfo {
        &self.nodes[id].output
    }

    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return node IDs in topological order.
    /// Since nodes are added in dependency order, this is just 0..n.
    pub fn topological_order(&self) -> Vec<NodeId> {
        (0..self.nodes.len()).collect()
    }

    /// Validate the graph: check that all input references are valid
    /// and precede the node that uses them.
    pub fn validate(&self) -> Result<(), GraphError> {
        for node in &self.nodes {
            for &input_id in &node.inputs {
                if input_id >= node.id {
                    return Err(GraphError::ForwardReference {
                        node: node.id,
                        input: input_id,
                    });
                }
                if input_id >= self.nodes.len() {
                    return Err(GraphError::InvalidNodeReference {
                        node: node.id,
                        input: input_id,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Errors in graph construction or validation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum GraphError {
    #[error("node {node} references future node {input} (forward reference)")]
    ForwardReference { node: NodeId, input: NodeId },

    #[error("node {node} references non-existent node {input}")]
    InvalidNodeReference { node: NodeId, input: NodeId },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_empty_graph() {
        let graph = Graph::new("test");
        assert_eq!(graph.name, "test");
        assert!(graph.is_empty());
    }

    #[test]
    fn add_nodes_and_validate() {
        let mut graph = Graph::new("test_model");

        // Input tokens
        let input = graph.input("tokens", Shape::new(vec![1, 128]), DType::I32);

        // Embedding weight
        let embed_w = graph.load_weight(
            "model.embed_tokens.weight",
            Shape::new(vec![32000, 2048]),
            DType::F16,
        );

        // Embedding lookup
        let tid = graph.alloc_tensor_id();
        let embed = graph.add_node(
            Op::Embedding {
                vocab_size: 32000,
                embed_dim: 2048,
            },
            vec![input, embed_w],
            TensorInfo {
                id: tid,
                name: "embed_out".into(),
                shape: Shape::new(vec![1, 128, 2048]),
                dtype: DType::F16,
            },
        );

        assert_eq!(graph.len(), 3);
        assert_eq!(graph.node(embed).inputs, vec![input, embed_w]);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn validate_detects_forward_reference() {
        let mut graph = Graph::new("bad");
        let tid = graph.alloc_tensor_id();
        // Manually push a node that references a future node
        graph.nodes.push(Node {
            id: 0,
            op: Op::Add,
            inputs: vec![1], // references node 1 which doesn't exist yet
            output: TensorInfo {
                id: tid,
                name: "bad".into(),
                shape: Shape::new(vec![1]),
                dtype: DType::F32,
            },
        });
        graph.next_node_id = 1;

        assert!(graph.validate().is_err());
    }

    #[test]
    fn shape_operations() {
        let s = Shape::new(vec![2, 3, 4]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.dim(1), 3);
        assert_eq!(s.to_string(), "[2, 3, 4]");
    }

    #[test]
    fn dtype_properties() {
        assert!(DType::Q4_0.is_quantized());
        assert!(!DType::F32.is_quantized());
        assert!(DType::F16.is_float());
        assert!(!DType::I32.is_float());
        assert_eq!(DType::F32.size_bytes(), 4);
    }

    #[test]
    fn topological_order() {
        let mut graph = Graph::new("topo");
        let a = graph.input("a", Shape::new(vec![4]), DType::F32);
        let b = graph.input("b", Shape::new(vec![4]), DType::F32);
        let tid = graph.alloc_tensor_id();
        let _c = graph.add_node(
            Op::Add,
            vec![a, b],
            TensorInfo {
                id: tid,
                name: "c".into(),
                shape: Shape::new(vec![4]),
                dtype: DType::F32,
            },
        );
        assert_eq!(graph.topological_order(), vec![0, 1, 2]);
    }

    #[test]
    fn weight_registration() {
        let mut graph = Graph::new("weights");
        graph.register_weight(
            "layer.0.attention.wq.weight".into(),
            Shape::new(vec![2048, 2048]),
            DType::F16,
        );
        assert!(graph.weights.contains_key("layer.0.attention.wq.weight"));
        let info = &graph.weights["layer.0.attention.wq.weight"];
        assert_eq!(info.shape, Shape::new(vec![2048, 2048]));
        assert_eq!(info.dtype, DType::F16);
    }

    #[test]
    fn model_config_roundtrip() {
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

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.architecture, Architecture::Llama);
        assert_eq!(deserialized.hidden_size, 2048);
        assert_eq!(deserialized.num_kv_heads, 8);
    }

    #[test]
    fn mistral_architecture_roundtrip() {
        let config = ModelConfig {
            architecture: Architecture::Mistral,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 32000,
            max_seq_len: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            dtype: DType::F16,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: Some(4096),
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.architecture, Architecture::Mistral);
        assert_eq!(deserialized.sliding_window_size, Some(4096));
        assert!(!deserialized.qkv_bias);
    }

    #[test]
    fn qwen2_architecture_sets_qkv_bias() {
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
            rope_theta: 1_000_000.0,
            dtype: DType::BF16,
            lm_head_dtype: None,
            proj_dtypes: None,
            sliding_window_size: None,
            qkv_bias: true,
            hidden_activation: HiddenActivation::SiLU,
        };

        assert!(config.qkv_bias);
        assert_eq!(config.architecture, Architecture::Qwen2);
        assert_eq!(config.sliding_window_size, None);
    }

    #[test]
    fn graph_with_config() {
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

        let graph = Graph::new("llama-1b").with_config(config);
        assert!(graph.config.is_some());
        let cfg = graph.config.unwrap();
        assert_eq!(cfg.architecture, Architecture::Llama);
    }

    #[test]
    fn build_transformer_layer_fragment() {
        let mut graph = Graph::new("layer_test");
        let hidden = 2048;

        // Simulate: input -> RMSNorm -> Q projection (matmul)
        let input = graph.input(
            "hidden_states",
            Shape::new(vec![1, 128, hidden]),
            DType::F16,
        );

        let norm_w = graph.load_weight(
            "model.layers.0.input_layernorm.weight",
            Shape::new(vec![hidden]),
            DType::F16,
        );

        let tid1 = graph.alloc_tensor_id();
        let normed = graph.add_node(
            Op::RMSNorm { eps: 1e-5 },
            vec![input, norm_w],
            TensorInfo {
                id: tid1,
                name: "normed".into(),
                shape: Shape::new(vec![1, 128, hidden]),
                dtype: DType::F16,
            },
        );

        let q_weight = graph.load_weight(
            "model.layers.0.self_attn.q_proj.weight",
            Shape::new(vec![hidden, hidden]),
            DType::F16,
        );

        let tid2 = graph.alloc_tensor_id();
        let q_proj = graph.add_node(
            Op::MatMul,
            vec![normed, q_weight],
            TensorInfo {
                id: tid2,
                name: "q_proj".into(),
                shape: Shape::new(vec![1, 128, hidden]),
                dtype: DType::F16,
            },
        );

        assert_eq!(graph.len(), 5); // input, norm_w, normed, q_weight, q_proj
        assert_eq!(graph.node(q_proj).inputs, vec![normed, q_weight]);
        assert!(graph.validate().is_ok());
    }
}
