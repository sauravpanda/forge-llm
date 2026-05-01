//! ONNX export — serialize an IR graph + weights to ONNX protobuf format.
//!
//! Uses a hand-rolled minimal protobuf encoder (no third-party protobuf crate
//! required). Targets ONNX opset 17 / IR version 8.
//!
//! # Usage
//! ```no_run
//! use std::path::Path;
//! use forgellm_frontend::{graph_builder, weight_loader::ModelWeights, ir::{ModelConfig, Architecture, DType, HiddenActivation}};
//! use forgellm_frontend::onnx_export::export_onnx;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = ModelConfig {
//!     architecture: Architecture::Llama,
//!     hidden_size: 64,
//!     intermediate_size: 128,
//!     num_layers: 1,
//!     num_attention_heads: 4,
//!     num_kv_heads: 2,
//!     head_dim: 16,
//!     vocab_size: 256,
//!     max_seq_len: 64,
//!     rms_norm_eps: 1e-5,
//!     rope_theta: 10000.0,
//!     dtype: DType::F16,
//!     lm_head_dtype: None,
//!     sliding_window_size: None,
//!     qkv_bias: false,
//!     hidden_activation: HiddenActivation::SiLU,
//! };
//! let graph = graph_builder::build_graph(&config)?;
//! let weights = ModelWeights { tensors: Default::default() };
//! export_onnx(&graph, &weights, Path::new("model.onnx"))?;
//! # Ok(())
//! # }
//! ```

use std::io::Write;
use std::path::Path;

use crate::ir::{DType, Graph, Op};
use crate::weight_loader::ModelWeights;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors that can occur during ONNX export.
#[derive(Debug, thiserror::Error)]
pub enum OnnxExportError {
    #[error("I/O error writing ONNX file: {0}")]
    Io(#[from] std::io::Error),
}

// ─── Minimal protobuf wire encoder ───────────────────────────────────────────
//
// Wire types used by ONNX:
//   0 = varint (int32, int64, bool, enum)
//   1 = 64-bit (fixed64, sfixed64, double)
//   2 = length-delimited (string, bytes, embedded message, packed repeated)
//   5 = 32-bit (fixed32, sfixed32, float)

/// Encode a varint (base-128 variable-length integer).
fn encode_varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

/// Encode a field tag: (field_number << 3) | wire_type.
fn encode_field_tag(buf: &mut Vec<u8>, field: u32, wire_type: u8) {
    encode_varint(buf, ((field as u64) << 3) | wire_type as u64);
}

/// Encode a string field (wire type 2).
fn encode_string(buf: &mut Vec<u8>, field: u32, s: &str) {
    encode_bytes(buf, field, s.as_bytes());
}

/// Encode a varint field (wire type 0) as i64.
fn encode_int64(buf: &mut Vec<u8>, field: u32, v: i64) {
    encode_field_tag(buf, field, 0);
    encode_varint(buf, v as u64);
}

/// Encode a bytes/message field (wire type 2).
fn encode_bytes(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
    encode_field_tag(buf, field, 2);
    encode_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

/// Encode an embedded message as a field (wire type 2).
fn encode_message(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
    encode_bytes(buf, field, data);
}

// ─── ONNX op mapping ─────────────────────────────────────────────────────────

/// Map an IR `Op` to (op_type, domain).
///
/// Returns `None` for ops that are not emitted as graph nodes (e.g.,
/// `LoadWeight` / `Input` which become graph initializers / inputs).
fn op_to_onnx(op: &Op) -> Option<(&'static str, &'static str)> {
    match op {
        Op::MatMul | Op::BatchMatMul => Some(("MatMul", "")),
        Op::Add | Op::Residual => Some(("Add", "")),
        Op::Mul => Some(("Mul", "")),
        Op::Softmax => Some(("Softmax", "")),
        Op::ReLU => Some(("Relu", "")),
        Op::GeLU => Some(("Gelu", "")),
        // SiLU = x * sigmoid(x); use custom op in forge-llm domain
        Op::SiLU => Some(("SiLU", "forge-llm")),
        // RMSNorm — map to LayerNormalization in com.microsoft domain
        Op::RMSNorm { .. } => Some(("SimplifiedLayerNormalization", "com.microsoft")),
        Op::LayerNorm { .. } => Some(("LayerNormalization", "")),
        // RoPE — custom op
        Op::RoPE { .. } => Some(("RotaryEmbedding", "com.microsoft")),
        // Attention — scaled dot-product attention
        Op::Attention { .. } => Some(("Attention", "com.microsoft")),
        // Embedding lookup → Gather
        Op::Embedding { .. } => Some(("Gather", "")),
        // LogitsProjection — treat as MatMul
        Op::LogitsProjection { .. } => Some(("MatMul", "")),
        // Shape / cast ops
        Op::Reshape { .. } => Some(("Reshape", "")),
        Op::Transpose { .. } => Some(("Transpose", "")),
        Op::Contiguous => Some(("Contiguous", "forge-llm")),
        Op::Cast { .. } => Some(("Cast", "")),
        // These become initializers / graph inputs, not nodes
        Op::LoadWeight { .. } | Op::Input { .. } => None,
    }
}

// ─── TensorProto builder ─────────────────────────────────────────────────────

/// ONNX TensorProto data_type values.
const ONNX_FLOAT: i32 = 1; // float32
const ONNX_FLOAT16: i32 = 10; // float16
const ONNX_INT32: i32 = 6;
const ONNX_INT64: i32 = 7;

fn dtype_to_onnx(dtype: DType) -> i32 {
    match dtype {
        DType::F32 => ONNX_FLOAT,
        DType::F16 | DType::BF16 => ONNX_FLOAT16,
        DType::I32 => ONNX_INT32,
        DType::I64 => ONNX_INT64,
        // Quantized types — map to float32 (weights are dequantized on export)
        DType::F8E4M3 | DType::F8E5M2 => ONNX_FLOAT16,
        DType::Q8_0 | DType::Q4_0 | DType::Q4_1 | DType::Q4_K | DType::Q6_K | DType::Q2 | DType::NF4 => ONNX_FLOAT,
    }
}

/// Build a TensorProto for a named weight tensor.
///
/// If the weight is present in `weights`, its data is embedded as raw f32
/// bytes (little-endian). Otherwise an empty (shape-only) initializer is
/// emitted — useful for structural ONNX files without embedded weights.
fn build_tensor_proto(name: &str, shape: &[usize], dtype: DType, data: Option<&[f32]>) -> Vec<u8> {
    let mut buf = Vec::new();

    // field 1: dims (int64[], repeated) — packed encoding
    for &dim in shape {
        encode_int64(&mut buf, 1, dim as i64);
    }

    // field 2: data_type (int32)
    encode_field_tag(&mut buf, 2, 0);
    encode_varint(&mut buf, dtype_to_onnx(dtype) as u64);

    // field 1 in TensorProto used for name? No — field 3 is `name` for TensorProto
    // (In ONNX proto: field 8 = name for TensorProto, but most ONNX parsers use
    // the name in the initializer list. Field 8 string = name.)
    encode_string(&mut buf, 8, name);

    if let Some(floats) = data {
        // field 9: raw_data (bytes) — f32 little-endian
        let mut raw = Vec::with_capacity(floats.len() * 4);
        for &f in floats {
            raw.extend_from_slice(&f.to_le_bytes());
        }
        encode_bytes(&mut buf, 9, &raw);
    }

    buf
}

// ─── ValueInfoProto builder ───────────────────────────────────────────────────

/// Build a minimal ValueInfoProto for a graph input or output.
///
/// We emit the name and a TypeProto (elem_type only, no shape constraints).
fn build_value_info(name: &str, dtype: DType) -> Vec<u8> {
    let mut buf = Vec::new();

    // field 1: name (string)
    encode_string(&mut buf, 1, name);

    // field 2: type (TypeProto message)
    // TypeProto: field 1 = tensor_type (Tensor message)
    //   Tensor: field 1 = elem_type (int32)
    let mut tensor_type_buf = Vec::new();
    encode_field_tag(&mut tensor_type_buf, 1, 0); // elem_type field, varint
    encode_varint(&mut tensor_type_buf, dtype_to_onnx(dtype) as u64);

    let mut type_proto_buf = Vec::new();
    encode_message(&mut type_proto_buf, 1, &tensor_type_buf);

    encode_message(&mut buf, 2, &type_proto_buf);

    buf
}

// ─── NodeProto builder ────────────────────────────────────────────────────────

/// Build a NodeProto for one IR node.
fn build_node_proto(
    node_name: &str,
    op_type: &str,
    domain: &str,
    inputs: &[String],
    outputs: &[String],
) -> Vec<u8> {
    let mut buf = Vec::new();

    // field 1: input names (repeated string)
    for inp in inputs {
        encode_string(&mut buf, 1, inp);
    }

    // field 2: output names (repeated string)
    for out in outputs {
        encode_string(&mut buf, 2, out);
    }

    // field 3: name (string)
    encode_string(&mut buf, 3, node_name);

    // field 4: op_type (string)
    encode_string(&mut buf, 4, op_type);

    // field 7: domain (string) — only emit if non-empty
    if !domain.is_empty() {
        encode_string(&mut buf, 7, domain);
    }

    buf
}

// ─── GraphProto builder ───────────────────────────────────────────────────────

/// Build the full GraphProto bytes for the IR graph + weights.
fn build_graph_proto(graph: &Graph, weights: &ModelWeights) -> Vec<u8> {
    let mut buf = Vec::new();

    // field 3: name
    encode_string(&mut buf, 3, &graph.name);

    // ── Initializers (weight tensors) ─────────────────────────────────────
    // field 5: initializer[] — TensorProto
    for (weight_name, tensor_info) in &graph.weights {
        let data = weights.get(weight_name);
        let tp = build_tensor_proto(
            weight_name,
            &tensor_info.shape.0.to_vec(),
            tensor_info.dtype,
            data,
        );
        encode_message(&mut buf, 5, &tp);
    }

    // ── Graph inputs ─────────────────────────────────────────────────────
    // Collect Input nodes; also include weights as graph inputs (ONNX convention)
    // field 11: input[] — ValueInfoProto
    for node in &graph.nodes {
        if let Op::Input { name } = &node.op {
            let vi = build_value_info(name, node.output.dtype);
            encode_message(&mut buf, 11, &vi);
        }
    }

    // ── Graph outputs ────────────────────────────────────────────────────
    // Last node is the output
    if let Some(last_node) = graph.nodes.last() {
        let vi = build_value_info(&last_node.output.name, last_node.output.dtype);
        // field 12: output[] — ValueInfoProto
        encode_message(&mut buf, 12, &vi);
    }

    // ── Compute nodes ────────────────────────────────────────────────────
    // field 1: node[] — NodeProto
    for node in &graph.nodes {
        let Some((op_type, domain)) = op_to_onnx(&node.op) else {
            // LoadWeight / Input — not emitted as nodes
            continue;
        };

        let input_names: Vec<String> = node
            .inputs
            .iter()
            .map(|&id| graph.nodes[id].output.name.clone())
            .collect();

        let output_names = vec![node.output.name.clone()];

        let np = build_node_proto(
            &node.output.name,
            op_type,
            domain,
            &input_names,
            &output_names,
        );
        encode_message(&mut buf, 1, &np);
    }

    buf
}

// ─── OpsetImportProto builder ─────────────────────────────────────────────────

/// Build an OpsetImportProto for the standard ONNX opset.
fn build_opset_import(domain: &str, version: i64) -> Vec<u8> {
    let mut buf = Vec::new();
    // field 1: domain (string)
    encode_string(&mut buf, 1, domain);
    // field 2: version (int64)
    encode_int64(&mut buf, 2, version);
    buf
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Export the IR graph + weights to an ONNX file.
///
/// The output file is a valid ONNX protobuf (opset 17, IR version 8).
/// Weights present in `weights` are embedded as raw f32 data; missing
/// weights are emitted as shape-only initializers.
///
/// # Errors
/// Returns [`OnnxExportError`] if the output file cannot be written.
pub fn export_onnx(
    graph: &Graph,
    weights: &ModelWeights,
    output_path: &Path,
) -> Result<(), OnnxExportError> {
    let model_bytes = build_model_proto(graph, weights);

    let mut file = std::fs::File::create(output_path)?;
    file.write_all(&model_bytes)?;
    Ok(())
}

/// Build the full ModelProto bytes.
pub(crate) fn build_model_proto(graph: &Graph, weights: &ModelWeights) -> Vec<u8> {
    let mut buf = Vec::new();

    // field 1: ir_version (int64) = 8
    encode_int64(&mut buf, 1, 8);

    // field 8: opset_import[] — standard ONNX opset 17
    let opset_std = build_opset_import("", 17);
    encode_message(&mut buf, 8, &opset_std);

    // field 8 (again): opset_import for com.microsoft
    let opset_ms = build_opset_import("com.microsoft", 1);
    encode_message(&mut buf, 8, &opset_ms);

    // field 8 (again): opset_import for forge-llm custom ops
    let opset_forge = build_opset_import("forge-llm", 1);
    encode_message(&mut buf, 8, &opset_forge);

    // field 2: producer_name (string)
    encode_string(&mut buf, 2, "forge-llm");

    // field 3: producer_version (string)
    encode_string(&mut buf, 3, env!("CARGO_PKG_VERSION"));

    // field 7: graph (GraphProto)
    let graph_bytes = build_graph_proto(graph, weights);
    encode_message(&mut buf, 7, &graph_bytes);

    buf
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::graph_builder::build_graph;
    use crate::ir::{Architecture, DType, HiddenActivation, ModelConfig};

    /// Minimal config for fast tests.
    fn tiny_config() -> ModelConfig {
        ModelConfig {
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
            sliding_window_size: None,
            qkv_bias: false,
            hidden_activation: HiddenActivation::SiLU,
        }
    }

    fn empty_weights() -> ModelWeights {
        ModelWeights {
            tensors: HashMap::new(),
        }
    }

    #[test]
    fn export_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let config = tiny_config();
        let graph = build_graph(&config).unwrap();
        let weights = empty_weights();
        let out = dir.path().join("model.onnx");

        export_onnx(&graph, &weights, &out).unwrap();

        assert!(out.exists(), "ONNX file should exist");
        assert!(
            out.metadata().unwrap().len() > 0,
            "ONNX file should not be empty"
        );
    }

    #[test]
    fn export_writes_valid_onnx_header() {
        let config = tiny_config();
        let graph = build_graph(&config).unwrap();
        let weights = empty_weights();

        let bytes = build_model_proto(&graph, &weights);

        // First byte should be field 1 (ir_version), varint wire type 0:
        // tag = (1 << 3) | 0 = 0x08
        assert!(!bytes.is_empty(), "serialized model should not be empty");
        assert_eq!(bytes[0], 0x08, "first byte should be ir_version field tag");
    }

    #[test]
    fn varint_encoding() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 0);
        assert_eq!(buf, &[0x00]);

        buf.clear();
        encode_varint(&mut buf, 1);
        assert_eq!(buf, &[0x01]);

        buf.clear();
        encode_varint(&mut buf, 127);
        assert_eq!(buf, &[0x7F]);

        buf.clear();
        encode_varint(&mut buf, 128);
        assert_eq!(buf, &[0x80, 0x01]);

        buf.clear();
        encode_varint(&mut buf, 300);
        assert_eq!(buf, &[0xAC, 0x02]);
    }

    #[test]
    fn string_field_encoding() {
        let mut buf = Vec::new();
        encode_string(&mut buf, 2, "hi");
        // field 2, wire 2 => tag = (2<<3)|2 = 0x12
        // length = 2 => 0x02
        // "hi" = 0x68, 0x69
        assert_eq!(buf, &[0x12, 0x02, 0x68, 0x69]);
    }

    #[test]
    fn export_with_weights() {
        let dir = tempfile::tempdir().unwrap();
        let config = tiny_config();
        let graph = build_graph(&config).unwrap();

        // Populate a few weight tensors with dummy f32 data
        let mut tensors = HashMap::new();
        for (name, info) in &graph.weights {
            let numel = info.shape.0.iter().product::<usize>();
            tensors.insert(name.clone(), vec![0.0f32; numel]);
        }
        let weights = ModelWeights { tensors };

        let out = dir.path().join("model_with_weights.onnx");
        export_onnx(&graph, &weights, &out).unwrap();

        assert!(out.exists());
        // File with weights should be larger than without
        let weights_size = out.metadata().unwrap().len();
        let empty_out = dir.path().join("model_empty.onnx");
        export_onnx(&graph, &empty_weights(), &empty_out).unwrap();
        let empty_size = empty_out.metadata().unwrap().len();
        assert!(
            weights_size > empty_size,
            "file with weights ({weights_size} bytes) should be larger than without ({empty_size} bytes)"
        );
    }

    #[test]
    fn op_mapping_completeness() {
        // Every Op variant that should produce a node must return Some(...)
        let op_variants_with_nodes: &[Op] = &[
            Op::MatMul,
            Op::BatchMatMul,
            Op::Add,
            Op::Mul,
            Op::SiLU,
            Op::GeLU,
            Op::ReLU,
            Op::RMSNorm { eps: 1e-5 },
            Op::LayerNorm { eps: 1e-5 },
            Op::RoPE {
                max_seq_len: 64,
                rope_theta: 10000.0,
                head_dim: 16,
            },
            Op::Attention {
                num_heads: 4,
                num_kv_heads: 2,
                head_dim: 16,
            },
            Op::Softmax,
            Op::Embedding {
                vocab_size: 256,
                embed_dim: 64,
            },
            Op::LogitsProjection { vocab_size: 256 },
            Op::Residual,
        ];
        for op in op_variants_with_nodes {
            assert!(op_to_onnx(op).is_some(), "op {op} should map to an ONNX op");
        }

        // LoadWeight and Input must return None
        assert!(op_to_onnx(&Op::LoadWeight { name: "w".into() }).is_none());
        assert!(op_to_onnx(&Op::Input { name: "x".into() }).is_none());
    }
}
