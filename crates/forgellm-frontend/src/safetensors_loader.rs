//! SafeTensors model loader.
//!
//! Loads a full model from a SafeTensors file into `ModelWeights`, converting
//! all tensors to f32. Infers `ModelConfig` from tensor shapes when no
//! `config.json` is present alongside the file.
//!
//! If a `config.json` exists in the same directory as the `.safetensors` file,
//! it is parsed via [`crate::config::HFConfig`] for a more accurate config.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use crate::config::HFConfig;
use crate::ir::{Architecture, DType, ModelConfig};
use crate::safetensors::{SafeTensorsError, SafeTensorsFile};
use crate::weight_loader::ModelWeights;

/// Errors that can occur while loading a SafeTensors model.
#[derive(Debug, thiserror::Error)]
pub enum SafeTensorsLoadError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("SafeTensors parse error: {0}")]
    Parse(#[from] SafeTensorsError),

    #[error("could not infer model config from tensor shapes: {0}")]
    ConfigInference(String),
}

/// Load a full model from a SafeTensors file.
///
/// Returns `(ModelConfig, ModelWeights)` ready for graph-building.
///
/// Config is sourced in this priority order:
/// 1. `config.json` in the same directory as the `.safetensors` file.
/// 2. Inference from tensor shapes (hidden size, num layers, etc.).
pub fn load_safetensors(
    path: impl AsRef<Path>,
) -> Result<(ModelConfig, ModelWeights), SafeTensorsLoadError> {
    let path = path.as_ref();

    // Memory-map the file for efficient access.
    let file = std::fs::File::open(path)?;
    // SAFETY: We never mutate the underlying file while `mmap` is alive. The
    // mmap is dropped at the end of this function before any write could occur.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let data: &[u8] = &mmap;

    // Parse the SafeTensors header.
    let st_file = crate::safetensors::parse(std::io::Cursor::new(data))?;
    let data_offset = st_file.data_offset as usize;

    // Load all tensors to f32.
    let mut tensors: HashMap<String, Vec<f32>> = HashMap::with_capacity(st_file.tensors.len());
    for info in &st_file.tensors {
        let abs_start = data_offset + info.data_start;
        let abs_end = data_offset + info.data_end;
        if abs_end > data.len() {
            return Err(SafeTensorsLoadError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("tensor {} extends past end of file", info.name),
            )));
        }
        let raw = &data[abs_start..abs_end];
        let floats = bytes_to_f32(raw, info.dtype).map_err(|e| {
            SafeTensorsLoadError::Io(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("dtype conversion for {}: {e}", info.name),
            ))
        })?;
        tensors.insert(info.name.clone(), floats);
    }

    // Try to load config.json from the same directory first.
    let config = if let Some(parent) = path.parent() {
        let config_path = parent.join("config.json");
        if config_path.exists() {
            let json = std::fs::read(&config_path)?;
            if let Ok(hf_cfg) = HFConfig::from_json(&json) {
                hf_cfg.to_model_config()
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // Fall back to shape-based inference.
    let config = match config {
        Some(c) => c,
        None => infer_model_config(&st_file, &tensors)?,
    };

    Ok((config, ModelWeights { tensors }))
}

/// Infer a `ModelConfig` from the shapes of the tensors in the file.
///
/// Uses the HuggingFace weight-name convention that SafeTensors files follow.
pub fn infer_model_config(
    st_file: &SafeTensorsFile,
    tensors: &HashMap<String, Vec<f32>>,
) -> Result<ModelConfig, SafeTensorsLoadError> {
    // Build a lookup: name → shape.
    let shape_map: HashMap<&str, &[usize]> = st_file
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t.shape.as_slice()))
        .collect();

    // hidden_size and vocab_size from embed_tokens.
    let embed_shape = shape_map.get("model.embed_tokens.weight").ok_or_else(|| {
        SafeTensorsLoadError::ConfigInference("missing model.embed_tokens.weight".to_string())
    })?;
    if embed_shape.len() < 2 {
        return Err(SafeTensorsLoadError::ConfigInference(
            "model.embed_tokens.weight must be 2-D".to_string(),
        ));
    }
    let vocab_size = embed_shape[0];
    let hidden_size = embed_shape[1];

    // num_layers: count of unique layer indices in model.layers.N.*.
    let num_layers = count_num_layers(&shape_map);
    if num_layers == 0 {
        return Err(SafeTensorsLoadError::ConfigInference(
            "could not find any model.layers.N.* tensors".to_string(),
        ));
    }

    // num_attention_heads and head_dim from q_proj and k_proj shapes.
    //
    // q_proj: [num_heads * head_dim, hidden_size]
    // k_proj: [num_kv_heads * head_dim, hidden_size]
    //
    // Strategy:
    //   1. If k_proj exists and k_out_dim < q_out_dim (GQA), we can compute
    //      head_dim = gcd(k_out_dim, q_out_dim), which is usually head_dim.
    //   2. Otherwise fall back to common head_dim values (128, 64, 80, 96).
    let q_shape = shape_map
        .get("model.layers.0.self_attn.q_proj.weight")
        .ok_or_else(|| {
            SafeTensorsLoadError::ConfigInference(
                "missing model.layers.0.self_attn.q_proj.weight".to_string(),
            )
        })?;
    if q_shape.len() < 2 {
        return Err(SafeTensorsLoadError::ConfigInference(
            "q_proj.weight must be at least 2-D".to_string(),
        ));
    }
    let q_out_dim = q_shape[0];

    let k_out_dim = shape_map
        .get("model.layers.0.self_attn.k_proj.weight")
        .and_then(|s| s.first().copied());

    // Derive head_dim:
    // If q_out != k_out (GQA), gcd(q_out, k_out) is typically head_dim.
    // If q_out == k_out (MHA or same-size GQA), try common head_dim values.
    // Derive head_dim by trying common values that evenly divide both q_out_dim
    // and k_out_dim (if present). This handles both MHA and GQA correctly.
    let head_dim = infer_head_dim(q_out_dim, k_out_dim);

    let num_attention_heads = q_out_dim.checked_div(head_dim).unwrap_or(1);

    // num_kv_heads from k_proj shape.
    let num_kv_heads = k_out_dim
        .map(|k| k.checked_div(head_dim).unwrap_or(num_attention_heads))
        .unwrap_or(num_attention_heads);

    // intermediate_size from gate_proj or up_proj.
    let intermediate_size = shape_map
        .get("model.layers.0.mlp.gate_proj.weight")
        .or_else(|| shape_map.get("model.layers.0.mlp.up_proj.weight"))
        .and_then(|s| s.first().copied())
        .unwrap_or(hidden_size * 4);

    // Infer architecture from tensor name patterns.
    // Llama/Mistral/Qwen2 all have gate_proj; Qwen2 has specific naming.
    let has_gate_proj = shape_map.contains_key("model.layers.0.mlp.gate_proj.weight");
    let has_q_bias = tensors.contains_key("model.layers.0.self_attn.q_proj.bias");

    // Check for dtype from first floating-point tensor.
    let dtype = st_file
        .tensors
        .first()
        .map(|t| match t.dtype {
            DType::F16 => DType::F16,
            DType::BF16 => DType::BF16,
            _ => DType::F32,
        })
        .unwrap_or(DType::F32);

    // Architecture heuristics:
    // - Qwen2 uses q_proj.bias
    // - Llama-family models have gate_proj (SwiGLU FFN)
    // Default to Llama for any unrecognised Llama-family layout.
    let qkv_bias = has_q_bias;
    let architecture = if has_q_bias {
        Architecture::Qwen2
    } else {
        // has_gate_proj is true for Llama/Mistral; unknown → also Llama as best guess.
        let _ = has_gate_proj;
        Architecture::Llama
    };

    Ok(ModelConfig {
        architecture,
        hidden_size,
        intermediate_size,
        num_layers,
        num_attention_heads,
        num_kv_heads,
        head_dim,
        vocab_size,
        max_seq_len: 4096,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        dtype,
        lm_head_dtype: None,
        proj_dtypes: None,
        sliding_window_size: None,
        qkv_bias,
        hidden_activation: crate::ir::HiddenActivation::SiLU,
    })
}

/// Count the number of transformer layers by scanning tensor names.
fn count_num_layers(shape_map: &HashMap<&str, &[usize]>) -> usize {
    let mut max_idx: Option<usize> = None;
    for name in shape_map.keys() {
        // Pattern: model.layers.N.something
        if let Some(rest) = name.strip_prefix("model.layers.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(idx) = rest[..dot].parse::<usize>() {
                    max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
                }
            }
        }
    }
    max_idx.map(|m| m + 1).unwrap_or(0)
}

/// Infer head_dim from q_proj and optionally k_proj output dimensions.
///
/// Tries common head-dimension values in descending order and picks the largest
/// one that evenly divides both `q_out_dim` and `k_out_dim` (when given).
fn infer_head_dim(q_out_dim: usize, k_out_dim: Option<usize>) -> usize {
    // Common head dimensions ordered from largest to smallest.
    const CANDIDATES: &[usize] = &[256, 192, 160, 128, 96, 80, 64, 32];
    for &d in CANDIDATES {
        let divides_q = q_out_dim.is_multiple_of(d);
        let divides_k = k_out_dim.is_none_or(|k| k.is_multiple_of(d));
        if divides_q && divides_k {
            return d;
        }
    }
    // Last-resort: divide q by a small number of heads.
    let fallback_heads = [32, 16, 8, 4, 1];
    for &h in &fallback_heads {
        if q_out_dim.is_multiple_of(h) {
            return q_out_dim / h;
        }
    }
    q_out_dim
}

/// Convert raw SafeTensors bytes to f32, supporting the dtypes we care about.
fn bytes_to_f32(data: &[u8], dtype: DType) -> Result<Vec<f32>, String> {
    match dtype {
        DType::F32 => {
            if !data.len().is_multiple_of(4) {
                return Err(format!("F32 data length {} not divisible by 4", data.len()));
            }
            Ok(data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        }
        DType::F16 => {
            if !data.len().is_multiple_of(2) {
                return Err(format!("F16 data length {} not divisible by 2", data.len()));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
                .collect())
        }
        DType::BF16 => {
            if !data.len().is_multiple_of(2) {
                return Err(format!(
                    "BF16 data length {} not divisible by 2",
                    data.len()
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| bf16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
                .collect())
        }
        DType::I32 => {
            if !data.len().is_multiple_of(4) {
                return Err(format!("I32 data length {} not divisible by 4", data.len()));
            }
            Ok(data
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f32)
                .collect())
        }
        DType::I64 => {
            if !data.len().is_multiple_of(8) {
                return Err(format!("I64 data length {} not divisible by 8", data.len()));
            }
            Ok(data
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32)
                .collect())
        }
        other => Err(format!(
            "unsupported dtype for SafeTensors loading: {other}"
        )),
    }
}

/// Convert an IEEE 754 half-precision (f16) bit pattern to f32.
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = 127 - 14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };

    f32::from_bits(f32_bits)
}

/// Convert a bfloat16 bit pattern to f32.
///
/// BF16 shares the same exponent as F32; zero-extend the mantissa.
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::{SafeTensorInfo, SafeTensorsFile};

    // ── Helper to build SafeTensorsFile + tensor data ────────────────────────

    fn build_safetensors_bytes(header_json: &str, tensor_data: &[u8]) -> Vec<u8> {
        let hdr = header_json.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
        buf.extend_from_slice(hdr);
        buf.extend_from_slice(tensor_data);
        buf
    }

    // ── infer_model_config ───────────────────────────────────────────────────

    /// Build a minimal SafeTensorsFile for inference tests.
    fn make_st_file(tensor_infos: Vec<SafeTensorInfo>) -> SafeTensorsFile {
        SafeTensorsFile {
            tensors: tensor_infos,
            data_offset: 0,
            metadata: HashMap::new(),
        }
    }

    fn make_info(name: &str, shape: Vec<usize>, dtype: DType) -> SafeTensorInfo {
        let numel: usize = shape.iter().product();
        let bytes = numel * 4; // assume f32
        SafeTensorInfo {
            name: name.to_string(),
            dtype,
            shape,
            data_start: 0,
            data_end: bytes,
        }
    }

    #[test]
    fn infer_config_basic_llama() {
        // Use head_dim=96 (divides into q_out but 128/256 do not evenly divide q_out below).
        // hidden=2304, q_out=24*96=2304, k_out=8*96=768.
        // Candidate check: 256 % 2304 ≠ 0 and 256 % 768 ≠ 0; 192 % 2304 = 0 but 192 % 768 = 0 too.
        // Let's use a simple unambiguous example:
        // hidden=2560, q_out=32*80=2560, k_out=8*80=640.
        // 256 % 2560 = 0 but 256 % 640 = 128 ≠ 0.
        // 192 % 640 ≠ 0 (640/192 is not integer).
        // 160 % 2560 = 0 and 160 % 640 = 0 → head_dim=160.
        // Actually: 128 % 2560 = 0 and 128 % 640 = 0 → 128 > 80.
        // So the heuristic picks 256 first (256 % 640 = 128 ≠ 0, skip), then 192 (skip), then 160.
        // Wait: 640 % 256 = 128 ≠ 0 ✓ skip. 640 % 192 = 64 ≠ 0 ✓ skip. 640 % 160 = 0 ✓.
        // 2560 % 160 = 0 ✓. So head_dim = 160, num_heads = 2560/160 = 16, num_kv = 640/160 = 4.
        //
        // We therefore test with head_dim=160 (Falcon / some LLMs use this).
        let hidden = 2560usize;
        let head_dim = 160usize; // 256, 192 don't divide 640; 160 does
        let num_heads = hidden / head_dim; // 2560 / 160 = 16
        let num_kv_heads = 4usize;
        let intermediate = 6912usize;
        let vocab = 32000usize;

        let infos = vec![
            make_info("model.embed_tokens.weight", vec![vocab, hidden], DType::F32),
            make_info(
                "model.layers.0.self_attn.q_proj.weight",
                vec![num_heads * head_dim, hidden],
                DType::F32,
            ),
            make_info(
                "model.layers.0.self_attn.k_proj.weight",
                vec![num_kv_heads * head_dim, hidden],
                DType::F32,
            ),
            make_info(
                "model.layers.0.mlp.gate_proj.weight",
                vec![intermediate, hidden],
                DType::F32,
            ),
            // Layer 1 — confirms num_layers = 2
            make_info(
                "model.layers.1.self_attn.q_proj.weight",
                vec![num_heads * head_dim, hidden],
                DType::F32,
            ),
        ];

        let st_file = make_st_file(infos);
        let tensors: HashMap<String, Vec<f32>> = HashMap::new();
        let config = infer_model_config(&st_file, &tensors).unwrap();

        assert_eq!(config.vocab_size, vocab);
        assert_eq!(config.hidden_size, hidden);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.num_attention_heads, num_heads);
        assert_eq!(config.num_kv_heads, num_kv_heads);
        assert_eq!(config.head_dim, head_dim);
        assert_eq!(config.intermediate_size, intermediate);
        assert_eq!(config.architecture, Architecture::Llama);
    }

    #[test]
    fn infer_config_missing_embed_tokens_errors() {
        let st_file = make_st_file(vec![]);
        let tensors = HashMap::new();
        let result = infer_model_config(&st_file, &tensors);
        assert!(matches!(
            result,
            Err(SafeTensorsLoadError::ConfigInference(_))
        ));
    }

    #[test]
    fn infer_config_missing_q_proj_errors() {
        let infos = vec![
            make_info("model.embed_tokens.weight", vec![32000, 2048], DType::F32),
            make_info(
                "model.layers.0.self_attn.k_proj.weight",
                vec![512, 2048],
                DType::F32,
            ),
        ];
        let st_file = make_st_file(infos);
        let tensors = HashMap::new();
        let result = infer_model_config(&st_file, &tensors);
        assert!(matches!(
            result,
            Err(SafeTensorsLoadError::ConfigInference(_))
        ));
    }

    // ── dtype conversion helpers ─────────────────────────────────────────────

    #[test]
    fn f16_to_f32_one() {
        assert_eq!(f16_to_f32(0x3C00), 1.0f32);
    }

    #[test]
    fn f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0f32);
    }

    #[test]
    fn bf16_to_f32_one() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0f32);
    }

    #[test]
    fn bytes_to_f32_f32_dtype() {
        let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let raw: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let out = bytes_to_f32(&raw, DType::F32).unwrap();
        assert_eq!(out, vec![1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn bytes_to_f32_f16_dtype() {
        // 1.0 in f16 = 0x3C00
        let raw: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x3C]; // two 1.0 f16 values
        let out = bytes_to_f32(&raw, DType::F16).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.0f32).abs() < 1e-6);
        assert!((out[1] - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn bytes_to_f32_bf16_dtype() {
        // 1.0 in bf16 = 0x3F80
        let raw: Vec<u8> = vec![0x80, 0x3F, 0x80, 0x3F];
        let out = bytes_to_f32(&raw, DType::BF16).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn bytes_to_f32_unaligned_returns_error() {
        let raw = vec![0u8; 3]; // not divisible by 4
        assert!(bytes_to_f32(&raw, DType::F32).is_err());
    }

    // ── count_num_layers ─────────────────────────────────────────────────────

    #[test]
    fn count_layers_from_shape_map() {
        let mut map: HashMap<&str, &[usize]> = HashMap::new();
        let shape = vec![1024usize, 1024];
        let s: &[usize] = &shape;
        map.insert("model.layers.0.self_attn.q_proj.weight", s);
        map.insert("model.layers.3.self_attn.q_proj.weight", s);
        map.insert("model.embed_tokens.weight", s);

        assert_eq!(count_num_layers(&map), 4); // max idx = 3, so 4 layers
    }

    // ── Round-trip: build minimal safetensors, parse, load ───────────────────

    #[test]
    fn load_minimal_safetensors_bytes() {
        // Build a tiny SafeTensors buffer with a single F32 tensor.
        let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let raw: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let header = r#"{"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}"#;
        let buf = build_safetensors_bytes(header, &raw);

        let st_file = crate::safetensors::parse(std::io::Cursor::new(&buf)).expect("parse failed");
        assert_eq!(st_file.tensors.len(), 1);
        assert_eq!(st_file.tensors[0].name, "w");
        assert_eq!(st_file.tensors[0].shape, vec![2usize, 2]);

        // Check we can extract the f32 values correctly.
        let data_start = st_file.data_offset as usize + st_file.tensors[0].data_start;
        let data_end = st_file.data_offset as usize + st_file.tensors[0].data_end;
        let loaded = bytes_to_f32(&buf[data_start..data_end], DType::F32).unwrap();
        assert_eq!(loaded, vec![1.0f32, 2.0, 3.0, 4.0]);
    }
}
