//! LoRA adapter loading and weight merging.
//!
//! Loads LoRA weights from SafeTensors format and merges them into
//! base model weights at compile time: W_merged = W + (alpha/rank) * B @ A
//!
//! LoRA adapter keys follow the HuggingFace convention:
//! - `base_model.model.layers.N.self_attn.q_proj.lora_A.weight`
//! - `base_model.model.layers.N.self_attn.q_proj.lora_B.weight`
//! - `base_model.model.layers.N.self_attn.q_proj.alpha` (optional scalar)

use std::collections::HashMap;
use std::path::Path;

use crate::weight_loader::ModelWeights;

/// A LoRA adapter loaded from a SafeTensors file.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Map from base-model weight name to the LoRA layer update.
    /// Keys are HuggingFace-style weight names (e.g. `model.layers.0.self_attn.q_proj.weight`).
    pub adapters: HashMap<String, LoraLayer>,
}

/// A single LoRA low-rank update for one weight matrix.
#[derive(Debug, Clone)]
pub struct LoraLayer {
    /// A matrix — shape [rank, in_features], row-major.
    pub lora_a: Vec<f32>,
    /// B matrix — shape [out_features, rank], row-major.
    pub lora_b: Vec<f32>,
    /// Low-rank dimension.
    pub rank: usize,
    /// Scaling factor alpha (often equal to rank so alpha/rank = 1.0).
    pub alpha: f32,
    pub in_features: usize,
    pub out_features: usize,
}

impl LoraLayer {
    /// Compute the LoRA delta: `(alpha / rank) * B @ A`.
    ///
    /// Returns a flat `[out_features * in_features]` f32 buffer in row-major
    /// order — the same layout as the base weight matrix.
    pub fn compute_delta(&self) -> Vec<f32> {
        let scale = self.alpha / self.rank as f32;
        let mut delta = vec![0.0f32; self.out_features * self.in_features];
        // B @ A: result[i][j] = sum_k B[i][k] * A[k][j]
        for i in 0..self.out_features {
            for k in 0..self.rank {
                let b_ik = self.lora_b[i * self.rank + k];
                for j in 0..self.in_features {
                    delta[i * self.in_features + j] += b_ik * self.lora_a[k * self.in_features + j];
                }
            }
        }
        for v in delta.iter_mut() {
            *v *= scale;
        }
        delta
    }
}

/// Errors that can occur while loading or applying a LoRA adapter.
#[derive(Debug, thiserror::Error)]
pub enum LoraError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SafeTensors parse error: {0}")]
    Parse(String),

    #[error("missing lora_A for layer: {0}")]
    MissingLoraA(String),

    #[error("missing lora_B for layer: {0}")]
    MissingLoraB(String),
}

// ──────────────────────────────────────────────────────────────────────────────
// SafeTensors minimal parser
// ──────────────────────────────────────────────────────────────────────────────

/// Map returned by [`parse_safetensors`]: tensor name → (f32 values, shape).
type ParsedTensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// Parse a SafeTensors byte buffer into a map of tensor-name → (f32 data, shape).
///
/// Only F32, F16, and BF16 tensors are materialised; scalar alpha values stored
/// as F32 are also handled. This avoids adding the `safetensors` crate as a
/// dependency — the format is simple enough to parse with `serde_json`.
fn parse_safetensors(data: &[u8]) -> Result<ParsedTensors, LoraError> {
    if data.len() < 8 {
        return Err(LoraError::Parse(
            "file too small for SafeTensors header".into(),
        ));
    }

    // First 8 bytes: header length (u64 LE)
    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;

    let header_end = 8 + header_len;
    if header_end > data.len() {
        return Err(LoraError::Parse(format!(
            "header length {header_len} exceeds file size {}",
            data.len()
        )));
    }

    let header_json = &data[8..header_end];
    let tensor_data_base = header_end; // where the raw tensor bytes start

    let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(header_json)
        .map_err(|e| LoraError::Parse(format!("JSON parse error: {e}")))?;

    let mut result = HashMap::new();

    for (key, value) in &raw {
        if key == "__metadata__" {
            continue;
        }

        let dtype = value
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LoraError::Parse(format!("missing dtype for tensor {key}")))?;

        let shape: Vec<usize> = value
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| LoraError::Parse(format!("missing shape for tensor {key}")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| LoraError::Parse(format!("invalid shape element in {key}")))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let offsets = value
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| LoraError::Parse(format!("missing data_offsets for tensor {key}")))?;

        if offsets.len() != 2 {
            return Err(LoraError::Parse(format!(
                "data_offsets must have 2 elements for tensor {key}"
            )));
        }

        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| LoraError::Parse(format!("invalid data_offsets[0] for {key}")))?
            as usize;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| LoraError::Parse(format!("invalid data_offsets[1] for {key}")))?
            as usize;

        let abs_start = tensor_data_base + start;
        let abs_end = tensor_data_base + end;

        if abs_end > data.len() {
            return Err(LoraError::Parse(format!(
                "tensor {key} data range [{start},{end}) exceeds file size"
            )));
        }

        let raw_bytes = &data[abs_start..abs_end];
        let floats = bytes_to_f32(raw_bytes, dtype)
            .map_err(|e| LoraError::Parse(format!("dtype conversion for {key}: {e}")))?;

        result.insert(key.clone(), (floats, shape));
    }

    Ok(result)
}

/// Convert raw bytes to f32, supporting F32, F16, and BF16 layouts.
fn bytes_to_f32(data: &[u8], dtype: &str) -> Result<Vec<f32>, String> {
    match dtype {
        "F32" => {
            if !data.len().is_multiple_of(4) {
                return Err(format!("F32 data length {} not divisible by 4", data.len()));
            }
            Ok(data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        }
        "F16" => {
            if !data.len().is_multiple_of(2) {
                return Err(format!("F16 data length {} not divisible by 2", data.len()));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes(b.try_into().unwrap());
                    f16_to_f32(bits)
                })
                .collect())
        }
        "BF16" => {
            if !data.len().is_multiple_of(2) {
                return Err(format!(
                    "BF16 data length {} not divisible by 2",
                    data.len()
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes(b.try_into().unwrap());
                    bf16_to_f32(bits)
                })
                .collect())
        }
        other => Err(format!("unsupported dtype for LoRA: {other}")),
    }
}

/// Convert an IEEE 754 half-precision (f16) bit pattern to f32.
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    // Sign, exponent (5-bit), mantissa (10-bit)
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal: normalise
            let mut e = 127 - 14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        // Inf or NaN
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };

    f32::from_bits(f32_bits)
}

/// Convert a bfloat16 bit pattern to f32.
///
/// BF16 shares the same exponent as F32; just zero-extend the mantissa.
#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ──────────────────────────────────────────────────────────────────────────────
// Key normalisation
// ──────────────────────────────────────────────────────────────────────────────

/// Strip the HuggingFace PEFT/LoRA prefix from a tensor key to obtain the
/// canonical base-model weight name.
///
/// Input examples:
/// - `base_model.model.layers.0.self_attn.q_proj.lora_A.weight`
/// - `base_model.model.layers.0.self_attn.q_proj.lora_B.weight`
///
/// Output examples:
/// - `("model.layers.0.self_attn.q_proj", LoraKeyKind::A)`
/// - `("model.layers.0.self_attn.q_proj", LoraKeyKind::B)`
#[derive(Debug, PartialEq)]
enum LoraKeyKind {
    A,
    B,
    Alpha,
}

fn parse_lora_key(key: &str) -> Option<(String, LoraKeyKind)> {
    // Remove `base_model.model.` prefix (PEFT convention)
    let trimmed = if let Some(rest) = key.strip_prefix("base_model.model.") {
        rest
    } else {
        key
    };

    // Detect lora_A / lora_B suffix
    if let Some(base) = trimmed.strip_suffix(".lora_A.weight") {
        return Some((format!("model.{base}.weight"), LoraKeyKind::A));
    }
    if let Some(base) = trimmed.strip_suffix(".lora_B.weight") {
        return Some((format!("model.{base}.weight"), LoraKeyKind::B));
    }
    // Some adapters use lora_alpha as a per-layer scalar stored as a tensor
    if let Some(base) = trimmed.strip_suffix(".lora_alpha") {
        return Some((format!("model.{base}.weight"), LoraKeyKind::Alpha));
    }
    // Fallback: plain `alpha` suffix (without a layer path)
    if trimmed.ends_with(".alpha") {
        let base = trimmed.strip_suffix(".alpha").unwrap();
        return Some((format!("model.{base}.weight"), LoraKeyKind::Alpha));
    }

    None
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Load a LoRA adapter from a SafeTensors file.
///
/// Parses all `lora_A` / `lora_B` tensor pairs and assembles a [`LoraAdapter`].
/// Keys are normalised to HuggingFace base-model convention so they can be
/// matched against weights produced by the GGUF → HF name mapper.
pub fn load_lora(path: impl AsRef<Path>) -> Result<LoraAdapter, LoraError> {
    let data = std::fs::read(path.as_ref())?;
    load_lora_from_bytes(&data)
}

/// Load a LoRA adapter from an in-memory SafeTensors byte buffer.
///
/// This is the core implementation; exposed separately to enable tests that
/// construct synthetic SafeTensors buffers without touching the filesystem.
pub fn load_lora_from_bytes(data: &[u8]) -> Result<LoraAdapter, LoraError> {
    let tensors = parse_safetensors(data)?;

    // Collect A matrices, B matrices, and alpha values keyed by base weight name
    let mut a_map: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut b_map: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut alpha_map: HashMap<String, f32> = HashMap::new();

    for (key, (data_vec, shape)) in tensors {
        match parse_lora_key(&key) {
            Some((base_name, LoraKeyKind::A)) => {
                a_map.insert(base_name, (data_vec, shape));
            }
            Some((base_name, LoraKeyKind::B)) => {
                b_map.insert(base_name, (data_vec, shape));
            }
            Some((base_name, LoraKeyKind::Alpha)) => {
                // Alpha may be a scalar tensor [1] or a rank-0 scalar []
                if let Some(&v) = data_vec.first() {
                    alpha_map.insert(base_name, v);
                }
            }
            None => {
                // Unknown / non-LoRA tensor — skip silently
            }
        }
    }

    // Build LoraLayer for each base name that has both A and B matrices
    let mut adapters = HashMap::new();

    for (base_name, (a_data, a_shape)) in &a_map {
        let (b_data, b_shape) = b_map
            .get(base_name)
            .ok_or_else(|| LoraError::MissingLoraB(base_name.clone()))?;

        // A: [rank, in_features]
        if a_shape.len() != 2 {
            return Err(LoraError::Parse(format!(
                "lora_A for {base_name} must be 2-D, got {} dims",
                a_shape.len()
            )));
        }
        let rank = a_shape[0];
        let in_features = a_shape[1];

        // B: [out_features, rank]
        if b_shape.len() != 2 {
            return Err(LoraError::Parse(format!(
                "lora_B for {base_name} must be 2-D, got {} dims",
                b_shape.len()
            )));
        }
        let out_features = b_shape[0];

        // Validate rank consistency
        if b_shape[1] != rank {
            return Err(LoraError::Parse(format!(
                "rank mismatch for {base_name}: lora_A rank={rank}, lora_B inner dim={}",
                b_shape[1]
            )));
        }

        let alpha = alpha_map.get(base_name).copied().unwrap_or(rank as f32);

        adapters.insert(
            base_name.clone(),
            LoraLayer {
                lora_a: a_data.clone(),
                lora_b: b_data.clone(),
                rank,
                alpha,
                in_features,
                out_features,
            },
        );
    }

    // Sanity check: every B must have a corresponding A
    for base_name in b_map.keys() {
        if !a_map.contains_key(base_name) {
            return Err(LoraError::MissingLoraA(base_name.clone()));
        }
    }

    Ok(LoraAdapter { adapters })
}

/// Merge LoRA adapter weights into base model weights in-place.
///
/// For each layer in the adapter, computes `W += (alpha/rank) * B @ A` and
/// adds the delta directly to the corresponding tensor in `base`.
///
/// Layers present in the adapter but not in the base weights are silently
/// skipped (the base model may use a different naming convention or subset).
pub fn merge_lora(base: &mut ModelWeights, lora: &LoraAdapter) {
    for (weight_name, layer) in &lora.adapters {
        if let Some(base_weight) = base.tensors.get_mut(weight_name) {
            let delta = layer.compute_delta();
            // delta and base_weight must have the same number of elements
            let len = base_weight.len().min(delta.len());
            for (w, d) in base_weight[..len].iter_mut().zip(delta[..len].iter()) {
                *w += d;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LoraLayer::compute_delta ─────────────────────────────────────────────

    /// Identity LoRA: B = I, A = I, rank = 2, alpha = 2.0 → delta = I (scale = 1.0)
    #[test]
    fn compute_delta_identity() {
        let layer = LoraLayer {
            // A: [[1,0],[0,1]]
            lora_a: vec![1.0, 0.0, 0.0, 1.0],
            // B: [[1,0],[0,1]]
            lora_b: vec![1.0, 0.0, 0.0, 1.0],
            rank: 2,
            alpha: 2.0,
            in_features: 2,
            out_features: 2,
        };
        let delta = layer.compute_delta();
        // scale = 2/2 = 1.0; B@A = I; delta = I
        assert_eq!(delta, vec![1.0, 0.0, 0.0, 1.0]);
    }

    /// Known small matrix multiply: out=2, rank=2, in=3, alpha=4, rank=2 → scale=2
    #[test]
    fn compute_delta_known_values() {
        // B (2×2): [[1,2],[3,4]]
        // A (2×3): [[1,0,1],[0,1,0]]
        // B@A = [[1,2],[3,4]] @ [[1,0,1],[0,1,0]]
        //      = [[1*1+2*0, 1*0+2*1, 1*1+2*0],
        //         [3*1+4*0, 3*0+4*1, 3*1+4*0]]
        //      = [[1,2,1],[3,4,3]]
        // scale = 4/2 = 2.0 → delta = [[2,4,2],[6,8,6]]
        let layer = LoraLayer {
            lora_a: vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            lora_b: vec![1.0, 2.0, 3.0, 4.0],
            rank: 2,
            alpha: 4.0,
            in_features: 3,
            out_features: 2,
        };
        let delta = layer.compute_delta();
        assert_eq!(delta, vec![2.0, 4.0, 2.0, 6.0, 8.0, 6.0]);
    }

    /// Zero B matrix → zero delta regardless of A.
    #[test]
    fn compute_delta_zero_b() {
        let layer = LoraLayer {
            lora_a: vec![1.0, 2.0, 3.0, 4.0],
            lora_b: vec![0.0, 0.0, 0.0, 0.0],
            rank: 2,
            alpha: 1.0,
            in_features: 2,
            out_features: 2,
        };
        let delta = layer.compute_delta();
        assert_eq!(delta, vec![0.0, 0.0, 0.0, 0.0]);
    }

    // ── merge_lora ───────────────────────────────────────────────────────────

    #[test]
    fn merge_lora_applies_delta() {
        // Base weight: 2×2 identity
        let mut base = ModelWeights {
            tensors: {
                let mut m = HashMap::new();
                m.insert(
                    "model.layers.0.self_attn.q_proj.weight".to_string(),
                    vec![1.0f32, 0.0, 0.0, 1.0],
                );
                m
            },
        };

        // LoRA layer with delta = [[0.5, 0], [0, 0.5]] (scale = 1, B@A = diag(0.5))
        // B: [[0.5,0],[0,0.5]] A: [[1,0],[0,1]] → B@A = diag(0.5)
        let lora = LoraAdapter {
            adapters: {
                let mut m = HashMap::new();
                m.insert(
                    "model.layers.0.self_attn.q_proj.weight".to_string(),
                    LoraLayer {
                        lora_a: vec![1.0, 0.0, 0.0, 1.0],
                        lora_b: vec![0.5, 0.0, 0.0, 0.5],
                        rank: 2,
                        alpha: 2.0, // scale = 2/2 = 1
                        in_features: 2,
                        out_features: 2,
                    },
                );
                m
            },
        };

        merge_lora(&mut base, &lora);
        let w = &base.tensors["model.layers.0.self_attn.q_proj.weight"];
        // delta = [[0.5,0],[0,0.5]], base = I → result = [[1.5,0],[0,1.5]]
        assert_eq!(*w, vec![1.5f32, 0.0, 0.0, 1.5]);
    }

    #[test]
    fn merge_lora_skips_missing_base_weight() {
        let mut base = ModelWeights {
            tensors: HashMap::new(),
        };
        let lora = LoraAdapter {
            adapters: {
                let mut m = HashMap::new();
                m.insert(
                    "model.layers.0.self_attn.q_proj.weight".to_string(),
                    LoraLayer {
                        lora_a: vec![1.0],
                        lora_b: vec![1.0],
                        rank: 1,
                        alpha: 1.0,
                        in_features: 1,
                        out_features: 1,
                    },
                );
                m
            },
        };
        // Must not panic
        merge_lora(&mut base, &lora);
        assert!(base.tensors.is_empty());
    }

    // ── SafeTensors parsing ──────────────────────────────────────────────────

    /// Build a minimal SafeTensors byte buffer from a JSON header + raw float data.
    fn build_safetensors(header_json: &str, tensor_data: &[u8]) -> Vec<u8> {
        let header_bytes = header_json.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(header_bytes);
        buf.extend_from_slice(tensor_data);
        buf
    }

    #[test]
    fn parse_safetensors_f32() {
        // One F32 tensor: shape [2,2], 4 floats → 16 bytes
        let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let raw: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let header = r#"{"w": {"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let buf = build_safetensors(header, &raw);

        let tensors = parse_safetensors(&buf).unwrap();
        assert_eq!(tensors["w"].0, vec![1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(tensors["w"].1, vec![2usize, 2]);
    }

    #[test]
    fn parse_safetensors_bf16() {
        // BF16 representation of 1.0 = 0x3F80 → as u16 LE = [0x80, 0x3F]
        let one_bf16: u16 = 0x3F80_u16;
        let raw: Vec<u8> = vec![
            (one_bf16 & 0xff) as u8,
            (one_bf16 >> 8) as u8,
            (one_bf16 & 0xff) as u8,
            (one_bf16 >> 8) as u8,
        ];
        let header = r#"{"w": {"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        let buf = build_safetensors(header, &raw);

        let tensors = parse_safetensors(&buf).unwrap();
        assert_eq!(tensors["w"].0, vec![1.0f32, 1.0f32]);
    }

    #[test]
    fn parse_safetensors_metadata_skipped() {
        let header = r#"{"__metadata__":{"version":"1"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let raw: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
        let buf = build_safetensors(header, &raw);

        let tensors = parse_safetensors(&buf).unwrap();
        assert!(!tensors.contains_key("__metadata__"));
        assert!(tensors.contains_key("w"));
    }

    #[test]
    fn parse_safetensors_too_small() {
        let result = parse_safetensors(&[0u8; 4]);
        assert!(matches!(result, Err(LoraError::Parse(_))));
    }

    // ── Key name normalisation ───────────────────────────────────────────────

    #[test]
    fn parse_lora_key_peft_convention() {
        let key = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let (base, kind) = parse_lora_key(key).unwrap();
        assert_eq!(base, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(kind, LoraKeyKind::A);
    }

    #[test]
    fn parse_lora_key_b_matrix() {
        let key = "base_model.model.layers.0.self_attn.q_proj.lora_B.weight";
        let (base, kind) = parse_lora_key(key).unwrap();
        assert_eq!(base, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(kind, LoraKeyKind::B);
    }

    #[test]
    fn parse_lora_key_alpha_tensor() {
        let key = "base_model.model.layers.0.self_attn.q_proj.lora_alpha";
        let (base, kind) = parse_lora_key(key).unwrap();
        assert_eq!(base, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(kind, LoraKeyKind::Alpha);
    }

    #[test]
    fn parse_lora_key_unknown_returns_none() {
        assert!(parse_lora_key("model.embed_tokens.weight").is_none());
        assert!(parse_lora_key("something.else").is_none());
    }

    // ── load_lora_from_bytes ─────────────────────────────────────────────────

    #[test]
    fn load_lora_from_bytes_roundtrip() {
        // Build a minimal LoRA SafeTensors with one A and one B tensor
        // Layer: q_proj, rank=2, in=3, out=2
        let a_data: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // [2,3]
        let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0]; // [2,2]
        let alpha_data: Vec<f32> = vec![4.0]; // scalar

        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let alpha_bytes: Vec<u8> = alpha_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        // data layout: a | b | alpha
        let a_len = a_bytes.len();
        let b_len = b_bytes.len();
        let alpha_len = alpha_bytes.len();
        let b_start = a_len;
        let alpha_start = a_len + b_len;

        let header = format!(
            r#"{{
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {{"dtype":"F32","shape":[2,3],"data_offsets":[0,{a_len}]}},
                "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": {{"dtype":"F32","shape":[2,2],"data_offsets":[{b_start},{alpha_start}]}},
                "base_model.model.layers.0.self_attn.q_proj.lora_alpha":    {{"dtype":"F32","shape":[1],"data_offsets":[{alpha_start},{total}]}}
            }}"#,
            a_len = a_len,
            b_start = b_start,
            alpha_start = alpha_start,
            total = alpha_start + alpha_len,
        );

        let mut tensor_data = Vec::new();
        tensor_data.extend_from_slice(&a_bytes);
        tensor_data.extend_from_slice(&b_bytes);
        tensor_data.extend_from_slice(&alpha_bytes);

        let header_bytes = header.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(header_bytes);
        buf.extend_from_slice(&tensor_data);

        let adapter = load_lora_from_bytes(&buf).unwrap();
        assert_eq!(adapter.adapters.len(), 1);
        let layer = &adapter.adapters["model.layers.0.self_attn.q_proj.weight"];
        assert_eq!(layer.rank, 2);
        assert_eq!(layer.in_features, 3);
        assert_eq!(layer.out_features, 2);
        assert!((layer.alpha - 4.0).abs() < 1e-6);
        assert_eq!(layer.lora_a, a_data);
        assert_eq!(layer.lora_b, b_data);
    }

    // ── f16 / bf16 conversion helpers ────────────────────────────────────────

    #[test]
    fn f16_to_f32_one() {
        // 1.0 in f16 = 0x3C00
        assert_eq!(f16_to_f32(0x3C00), 1.0f32);
    }

    #[test]
    fn f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0f32);
    }

    #[test]
    fn bf16_to_f32_one() {
        // 1.0 in bf16 = 0x3F80
        assert_eq!(bf16_to_f32(0x3F80), 1.0f32);
    }
}
