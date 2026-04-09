# Intermediate Representation

The IR is the central data structure in ForgeLLM. It represents a static computation graph with typed tensor operations specialized for transformer architectures.

## Core Types

### DType

Supported data types:

- **Float**: F32, F16, BF16, F8E4M3, F8E5M2
- **Quantized**: Q8_0, Q4_0, Q4_1, Q2, NF4
- **Integer**: I32, I64

### Shape

A list of dimension sizes. Shapes are fully known at compile time — there are no dynamic dimensions.

### Op

Operations cover the full transformer vocabulary:

| Category | Operations |
|----------|-----------|
| Creation | `LoadWeight`, `Input` |
| Linear Algebra | `MatMul`, `BatchMatMul` |
| Elementwise | `Add`, `Mul`, `SiLU`, `GeLU`, `ReLU` |
| Normalization | `RMSNorm`, `LayerNorm` |
| Attention | `RoPE`, `Attention`, `Softmax` |
| Shape | `Reshape`, `Transpose`, `Contiguous` |
| Model | `Embedding`, `LogitsProjection`, `Residual` |
| Cast | `Cast` |

### Graph

The computation graph is a DAG of `Node`s. Each node has:
- An `Op` (what to compute)
- Input node IDs (data dependencies)
- Output `TensorInfo` (shape + dtype of the result)

Nodes are stored in topological order — each node's inputs always have lower IDs.

### ModelConfig

Hyperparameters for the model:
- Architecture (Llama, Qwen2, Mistral, etc.)
- Dimensions (hidden_size, intermediate_size, num_layers, etc.)
- Normalization parameters (rms_norm_eps)
- Position encoding (rope_theta, max_seq_len)

## Graph Construction

The `graph_builder` module constructs a complete forward-pass graph from a `ModelConfig`. For a Llama model, this includes:

1. Token embedding lookup
2. For each layer:
   - Input LayerNorm (RMSNorm)
   - Q/K/V projections (3 MatMuls)
   - RoPE on Q and K
   - Grouped-query attention
   - Output projection (MatMul)
   - Residual connection
   - Post-attention LayerNorm (RMSNorm)
   - SiLU-gated FFN (gate, up, down projections)
   - Residual connection
3. Final LayerNorm
4. Logits projection
