//! Weight loader — reads and dequantizes tensor data from GGUF files.
//!
//! Loads tensor data into f32 buffers, handling dequantization for
//! quantized formats (Q4_0, Q8_0, etc.) and fp16→f32 conversion.

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use crate::gguf::{self, GGMLType, GGUFFile, GGUFTensorInfo};

/// Raw weight data — either f32 (dequantized), Q8_0 raw bytes, or Q4_0 raw bytes.
#[derive(Debug, Clone)]
pub enum WeightData {
    /// Fully dequantized f32 tensor.
    F32(Vec<f32>),
    /// Raw Q8_0 bytes: N_blocks * 34 bytes each block = [2 bytes f16 scale][32 bytes int8].
    Q8_0Raw(Vec<u8>),
    /// Raw Q4_0 bytes: N_blocks * 18 bytes each block = [2 bytes f16 scale][16 bytes 4-bit pairs].
    Q4_0Raw(Vec<u8>),
    /// Raw Q4_K bytes: N_blocks * 144 bytes each super-block of 256 elements.
    Q4_KRaw(Vec<u8>),
}

/// Model weights with mixed storage: Q8_0 kept as raw bytes, others as f32.
#[derive(Debug, Clone)]
pub struct ModelWeightsRaw {
    pub tensors: HashMap<String, WeightData>,
}

impl ModelWeightsRaw {
    /// Get an f32 tensor by name.
    pub fn get_f32(&self, name: &str) -> Option<&[f32]> {
        match self.tensors.get(name) {
            Some(WeightData::F32(v)) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Get a Q8_0 raw byte tensor by name.
    pub fn get_q8_raw(&self, name: &str) -> Option<&[u8]> {
        match self.tensors.get(name) {
            Some(WeightData::Q8_0Raw(v)) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Get a Q4_0 raw byte tensor by name.
    pub fn get_q4_raw(&self, name: &str) -> Option<&[u8]> {
        match self.tensors.get(name) {
            Some(WeightData::Q4_0Raw(v)) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Get a Q4_K raw byte tensor by name.
    pub fn get_q4k_raw(&self, name: &str) -> Option<&[u8]> {
        match self.tensors.get(name) {
            Some(WeightData::Q4_KRaw(v)) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether no tensors are loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.tensors
            .values()
            .map(|v| match v {
                WeightData::F32(f) => f.len() * 4,
                WeightData::Q8_0Raw(b) => b.len(),
                WeightData::Q4_0Raw(b) => b.len(),
                WeightData::Q4_KRaw(b) => b.len(),
            })
            .sum()
    }

    /// Get a tensor by name regardless of type — for non-Q8_0 tensors only.
    pub fn get(&self, name: &str) -> Option<&WeightData> {
        self.tensors.get(name)
    }
}

/// Load all tensors from a GGUF file with mixed storage.  See
/// `load_from_file_mixed_with_target` for the full behaviour; this entry
/// point selects the default Q8_0 routing for non-native K-quants.
pub fn load_from_file_mixed(
    path: impl AsRef<std::path::Path>,
) -> Result<(crate::gguf::GGUFFile, ModelWeightsRaw), WeightLoadError> {
    load_from_file_mixed_with_target(path, None)
}

/// Load all tensors from a GGUF file with mixed storage and an optional
/// target dtype hint.
///
/// Native paths (always):
/// - Q8_0 tensors are kept as raw bytes (`WeightData::Q8_0Raw`).
/// - Q4_0 tensors are kept as raw bytes (`WeightData::Q4_0Raw`).
/// - F32 / F16 / BF16 are dequantized to f32 (`WeightData::F32`).
///
/// Other quantized formats are re-quantized to a single uniform target so
/// the codegen sees a consistent layout per projection tensor:
/// - `target_dtype = None` or `Some(DType::Q8_0)` (default): dequant → Q8_0
///   for everything else.  Mixed Q4_K_M files (Q4_K + Q6_K per layer) all
///   land on the Q8_0 code path.
/// - `target_dtype = Some(DType::Q4_K)`: Q4_K stays raw (`Q4_KRaw`); other
///   K-quants (Q5_K, Q6_K, Q5_0/1, ...) dequant → Q4_K via
///   `quantize_f32_to_q4_k` so the whole model fits the Q4_K kernel.
pub fn load_from_file_mixed_with_target(
    path: impl AsRef<std::path::Path>,
    target_dtype: Option<crate::ir::DType>,
) -> Result<(crate::gguf::GGUFFile, ModelWeightsRaw), WeightLoadError> {
    use crate::ir::DType;
    let file = std::fs::File::open(path.as_ref())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let mut cursor = std::io::Cursor::new(&mmap[..]);

    let gguf = crate::gguf::parse(&mut cursor)
        .map_err(|e| WeightLoadError::Io(std::io::Error::other(e)))?;

    let want_q4k = target_dtype == Some(DType::Q4_K);

    let mut tensors = HashMap::with_capacity(gguf.tensors.len());
    for tensor_info in &gguf.tensors {
        let data_offset = gguf.tensor_data_offset + tensor_info.offset;
        let data_size = tensor_info.data_size() as usize;
        let numel = tensor_info.numel() as usize;

        let start = data_offset as usize;
        let end = start + data_size;
        if end > mmap.len() {
            return Err(WeightLoadError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("tensor {} extends past end of file", tensor_info.name),
            )));
        }

        let raw = &mmap[start..end];
        let hf_name = gguf_name_to_hf(&tensor_info.name);

        let weight_data = match tensor_info.ggml_type {
            GGMLType::Q8_0 => WeightData::Q8_0Raw(raw.to_vec()),
            GGMLType::Q4_0 => WeightData::Q4_0Raw(raw.to_vec()),
            // Dense / non-quantized: dequant to f32 and store dense.
            GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 => {
                let f32_data = dequantize(raw, tensor_info.ggml_type, numel)?;
                WeightData::F32(f32_data)
            }
            // Q4_K target: keep Q4_K raw; requant other K-quants to Q4_K.
            GGMLType::Q4K if want_q4k => WeightData::Q4_KRaw(raw.to_vec()),
            _ if want_q4k && numel.is_multiple_of(256) => {
                // Other quantized format under a Q4_K target — requant via f32.
                let f32_data = dequantize(raw, tensor_info.ggml_type, numel)?;
                WeightData::Q4_KRaw(quantize_f32_to_q4_k(&f32_data))
            }
            // Default (Q8_0) target, or Q4_K target with non-multiple-of-256
            // numel: dequant → Q8_0 for everything else.
            _ => {
                let f32_data = dequantize(raw, tensor_info.ggml_type, numel)?;
                WeightData::Q8_0Raw(quantize_f32_to_q8_0(&f32_data))
            }
        };

        tensors.insert(hf_name, weight_data);
    }

    let mut weights = ModelWeightsRaw { tensors };
    auto_split_fused_tensors_raw(&mut weights, &gguf);
    Ok((gguf, weights))
}

/// Loaded model weights — all tensors dequantized to f32.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Map from tensor name to f32 data.
    pub tensors: HashMap<String, Vec<f32>>,
}

impl ModelWeights {
    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|v| v.as_slice())
    }

    /// Get a tensor by name, panicking if not found.
    pub fn tensor(&self, name: &str) -> &[f32] {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("weight not found: {name}"))
    }

    /// Number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether no tensors are loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Total number of f32 elements across all tensors.
    pub fn total_elements(&self) -> usize {
        self.tensors.values().map(|v| v.len()).sum()
    }

    /// Total memory usage in bytes (f32 only).
    pub fn memory_bytes(&self) -> usize {
        self.total_elements() * 4
    }
}

/// Apply Gemma-1's RMS-norm `+1` offset in place (loaded f32 weights).
///
/// Gemma-1's RMS norm formula is `x * (1 + w) / rsqrt(...)` rather than the
/// standard `x * w / rsqrt(...)`. Equivalent to storing `w' = w + 1` and then
/// using the standard formula, so kernels need no new code path.
///
/// Applied to every `*.input_layernorm.weight`, `*.post_attention_layernorm.weight`,
/// and `model.norm.weight` tensor present in the map.
///
/// **IMPORTANT:** standard `llama.cpp`-converted Gemma GGUFs already have the
/// `+1` offset baked into the stored weights (see `convert_hf_to_gguf.py`:
/// `data_torch = data_torch + 1` for `.norm.weight` tensors). Do **not** call
/// this helper on a GGUF-loaded `ModelWeights` — you would double-offset.
///
/// This helper is intended for raw HuggingFace / Safetensors weights, where
/// the `+1` is normally applied inside the reference RMSNorm forward pass.
/// No call site inside ForgeLLM uses it automatically today; it is exported
/// for future Safetensors-origin Gemma support.
///
/// Embedding scale (`sqrt(hidden_size)`) is **not** handled here because
/// `model.embed_tokens.weight` may be tied to `lm_head.weight`; the scale is
/// applied at embed-lookup time in the interpreter instead.
pub fn apply_gemma_weight_tweaks(
    weights: &mut ModelWeights,
    _hidden_size: usize,
    num_layers: usize,
) {
    for layer in 0..num_layers {
        for name in [
            format!("model.layers.{layer}.input_layernorm.weight"),
            format!("model.layers.{layer}.post_attention_layernorm.weight"),
        ] {
            if let Some(w) = weights.tensors.get_mut(&name) {
                for v in w.iter_mut() {
                    *v += 1.0;
                }
            }
        }
    }
    if let Some(w) = weights.tensors.get_mut("model.norm.weight") {
        for v in w.iter_mut() {
            *v += 1.0;
        }
    }
}

/// Errors during weight loading.
#[derive(Debug, thiserror::Error)]
pub enum WeightLoadError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("tensor not found in GGUF: {0}")]
    TensorNotFound(String),

    #[error("unsupported GGML type for dequantization: {0:?}")]
    UnsupportedType(GGMLType),
}

/// Load all tensors from a GGUF file, dequantizing to f32.
/// Tensor names are remapped from GGUF convention to HuggingFace convention.
pub fn load_all<R: Read + Seek>(
    reader: &mut R,
    gguf: &GGUFFile,
) -> Result<ModelWeights, WeightLoadError> {
    let mut tensors = HashMap::with_capacity(gguf.tensors.len());

    for tensor_info in &gguf.tensors {
        let data = load_tensor(reader, gguf, tensor_info)?;
        let hf_name = gguf_name_to_hf(&tensor_info.name);
        tensors.insert(hf_name, data);
    }

    let mut weights = ModelWeights { tensors };
    auto_split_fused_tensors_f32(&mut weights, gguf);
    Ok(weights)
}

/// Load all tensors from a GGUF file using memory mapping.
///
/// This is faster than `load_all` because it avoids reading the entire
/// file into a Vec first — the OS maps it directly into virtual memory.
pub fn load_from_file(path: impl AsRef<Path>) -> Result<(GGUFFile, ModelWeights), WeightLoadError> {
    let file = std::fs::File::open(path.as_ref())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let mut cursor = io::Cursor::new(&mmap[..]);

    let gguf = gguf::parse(&mut cursor).map_err(|e| WeightLoadError::Io(io::Error::other(e)))?;

    let mut tensors = HashMap::with_capacity(gguf.tensors.len());
    for tensor_info in &gguf.tensors {
        let data_offset = gguf.tensor_data_offset + tensor_info.offset;
        let data_size = tensor_info.data_size() as usize;
        let numel = tensor_info.numel() as usize;

        let start = data_offset as usize;
        let end = start + data_size;
        if end > mmap.len() {
            return Err(WeightLoadError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("tensor {} extends past end of file", tensor_info.name),
            )));
        }

        let raw = &mmap[start..end];
        let data = dequantize(raw, tensor_info.ggml_type, numel)?;
        let hf_name = gguf_name_to_hf(&tensor_info.name);
        tensors.insert(hf_name, data);
    }

    let mut weights = ModelWeights { tensors };
    auto_split_fused_tensors_f32(&mut weights, &gguf);
    Ok((gguf, weights))
}

/// Remap GGUF tensor names to HuggingFace convention.
///
/// GGUF: `token_embd.weight`, `blk.0.attn_q.weight`, `output_norm.weight`
/// HF:   `model.embed_tokens.weight`, `model.layers.0.self_attn.q_proj.weight`, `model.norm.weight`
fn gguf_name_to_hf(name: &str) -> String {
    // Direct mappings for non-layer tensors
    match name {
        "token_embd.weight" => return "model.embed_tokens.weight".to_string(),
        "output_norm.weight" => return "model.norm.weight".to_string(),
        "output.weight" => return "lm_head.weight".to_string(),
        _ => {}
    }

    // Layer tensor mappings: blk.N.xxx → model.layers.N.yyy
    if let Some(rest) = name.strip_prefix("blk.") {
        // Parse layer number
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];

            let hf_suffix = match suffix {
                "attn_norm.weight" => "input_layernorm.weight",
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_q.bias" => "self_attn.q_proj.bias",
                "attn_k.bias" => "self_attn.k_proj.bias",
                "attn_v.bias" => "self_attn.v_proj.bias",
                "attn_output.weight" => "self_attn.o_proj.weight",
                // Phi-3: Q/K/V concatenated along the output dim in one tensor.
                // Split post-load via `split_fused_tensors`.
                "attn_qkv.weight" => "self_attn.qkv_proj.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                other => other, // Pass through unknown suffixes
            };

            return format!("model.layers.{layer_num}.{hf_suffix}");
        }
    }

    // Fallback: return original name
    name.to_string()
}

/// Auto-detect a fused Phi-3-style layout from the loaded weight map and
/// the GGUF metadata, then split in place. No-op when neither fused
/// entry is present (standard Llama/Qwen2/Gemma layouts).
fn auto_split_fused_tensors_raw(weights: &mut ModelWeightsRaw, gguf: &GGUFFile) {
    let has_fused_qkv = weights
        .tensors
        .contains_key("model.layers.0.self_attn.qkv_proj.weight");
    let has_fused_ffn = {
        let gate = "model.layers.0.mlp.gate_proj.weight";
        let up = "model.layers.0.mlp.up_proj.weight";
        !weights.tensors.contains_key(gate) && weights.tensors.contains_key(up)
    };
    if !has_fused_qkv && !has_fused_ffn {
        return;
    }
    if let Some(params) = fused_params_from_gguf(gguf) {
        split_fused_tensors(
            weights,
            params.num_layers,
            params.hidden_size,
            params.num_heads,
            params.num_kv_heads,
            params.head_dim,
            params.intermediate_size,
        );
    }
}

fn auto_split_fused_tensors_f32(weights: &mut ModelWeights, gguf: &GGUFFile) {
    let has_fused_qkv = weights
        .tensors
        .contains_key("model.layers.0.self_attn.qkv_proj.weight");
    let has_fused_ffn = {
        let gate = "model.layers.0.mlp.gate_proj.weight";
        let up = "model.layers.0.mlp.up_proj.weight";
        !weights.tensors.contains_key(gate) && weights.tensors.contains_key(up)
    };
    if !has_fused_qkv && !has_fused_ffn {
        return;
    }
    if let Some(params) = fused_params_from_gguf(gguf) {
        split_fused_tensors_f32(
            weights,
            params.num_layers,
            params.hidden_size,
            params.num_heads,
            params.num_kv_heads,
            params.head_dim,
            params.intermediate_size,
        );
    }
}

struct FusedParams {
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
}

fn fused_params_from_gguf(gguf: &GGUFFile) -> Option<FusedParams> {
    let arch = gguf.get_str("general.architecture")?.to_string();
    let num_layers = gguf.get_u32(&format!("{arch}.block_count"))? as usize;
    let hidden_size = gguf.get_u32(&format!("{arch}.embedding_length"))? as usize;
    let num_heads = gguf.get_u32(&format!("{arch}.attention.head_count"))? as usize;
    let num_kv_heads = gguf
        .get_u32(&format!("{arch}.attention.head_count_kv"))
        .map(|v| v as usize)
        .unwrap_or(num_heads);
    let intermediate_size = gguf
        .get_u32(&format!("{arch}.feed_forward_length"))
        .map(|v| v as usize)
        .unwrap_or(hidden_size * 4);
    let head_dim = hidden_size / num_heads;
    Some(FusedParams {
        num_layers,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
    })
}

/// Split fused Phi-3-style tensors in place.
///
/// Phi-3 GGUFs store attention as one fused `attn_qkv.weight` (Q|K|V
/// concatenated along the output dim) and MLP as one fused `ffn_up.weight`
/// (gate|up concatenated). The rest of the pipeline expects HF-split
/// `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj`; this helper
/// rewrites the weight map to that shape. No-op when the fused entries
/// are not present, so it's safe to call for any architecture.
///
/// GGUF byte layout for a tensor with dims=(in_features, out_features)
/// places the out dim as the outer (slow) index, so concatenated Q|K|V
/// are contiguous byte ranges. For Q8_0 each section must be block-aligned
/// (multiple of 32 elements), which holds for all current Phi-3 shapes
/// (hidden × Q/K/V rows, where rows is a multiple of 32).
pub fn split_fused_tensors(
    weights: &mut ModelWeightsRaw,
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
) {
    for layer in 0..num_layers {
        let qkv_name = format!("model.layers.{layer}.self_attn.qkv_proj.weight");
        if let Some(fused) = weights.tensors.remove(&qkv_name) {
            let q_elems = hidden_size * num_heads * head_dim;
            let kv_elems = hidden_size * num_kv_heads * head_dim;
            let (q, k, v) = split_weight_three(&fused, q_elems, kv_elems, kv_elems);
            weights
                .tensors
                .insert(format!("model.layers.{layer}.self_attn.q_proj.weight"), q);
            weights
                .tensors
                .insert(format!("model.layers.{layer}.self_attn.k_proj.weight"), k);
            weights
                .tensors
                .insert(format!("model.layers.{layer}.self_attn.v_proj.weight"), v);
        }

        let gate_name = format!("model.layers.{layer}.mlp.gate_proj.weight");
        let up_name = format!("model.layers.{layer}.mlp.up_proj.weight");
        if !weights.tensors.contains_key(&gate_name) {
            let fused_size = weights.tensors.get(&up_name).map(weight_elem_count);
            let expected_single = hidden_size * intermediate_size;
            if fused_size == Some(2 * expected_single) {
                let fused = weights.tensors.remove(&up_name).unwrap();
                let (gate, up) = split_weight_two(&fused, expected_single, expected_single);
                weights.tensors.insert(gate_name, gate);
                weights.tensors.insert(up_name, up);
            }
        }
    }
}

/// Same as [`split_fused_tensors`] but operates on fully-dequantized
/// [`ModelWeights`]. Used by the interpreter path where everything is f32.
pub fn split_fused_tensors_f32(
    weights: &mut ModelWeights,
    num_layers: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
) {
    for layer in 0..num_layers {
        let qkv_name = format!("model.layers.{layer}.self_attn.qkv_proj.weight");
        if let Some(fused) = weights.tensors.remove(&qkv_name) {
            let q_elems = hidden_size * num_heads * head_dim;
            let kv_elems = hidden_size * num_kv_heads * head_dim;
            assert_eq!(fused.len(), q_elems + 2 * kv_elems);
            let q = fused[0..q_elems].to_vec();
            let k = fused[q_elems..q_elems + kv_elems].to_vec();
            let v = fused[q_elems + kv_elems..].to_vec();
            weights
                .tensors
                .insert(format!("model.layers.{layer}.self_attn.q_proj.weight"), q);
            weights
                .tensors
                .insert(format!("model.layers.{layer}.self_attn.k_proj.weight"), k);
            weights
                .tensors
                .insert(format!("model.layers.{layer}.self_attn.v_proj.weight"), v);
        }

        let gate_name = format!("model.layers.{layer}.mlp.gate_proj.weight");
        let up_name = format!("model.layers.{layer}.mlp.up_proj.weight");
        if !weights.tensors.contains_key(&gate_name) {
            let fused_size = weights.tensors.get(&up_name).map(|v| v.len());
            let expected_single = hidden_size * intermediate_size;
            if fused_size == Some(2 * expected_single) {
                let fused = weights.tensors.remove(&up_name).unwrap();
                let gate = fused[0..expected_single].to_vec();
                let up = fused[expected_single..].to_vec();
                weights.tensors.insert(gate_name, gate);
                weights.tensors.insert(up_name, up);
            }
        }
    }
}

fn weight_elem_count(w: &WeightData) -> usize {
    match w {
        WeightData::F32(v) => v.len(),
        WeightData::Q8_0Raw(b) => b.len() / 34 * 32,
        WeightData::Q4_0Raw(b) => b.len() / 18 * 32,
        WeightData::Q4_KRaw(b) => b.len() / 144 * 256,
    }
}

fn split_weight_three(
    w: &WeightData,
    n1: usize,
    n2: usize,
    n3: usize,
) -> (WeightData, WeightData, WeightData) {
    match w {
        WeightData::F32(v) => {
            assert_eq!(v.len(), n1 + n2 + n3);
            (
                WeightData::F32(v[0..n1].to_vec()),
                WeightData::F32(v[n1..n1 + n2].to_vec()),
                WeightData::F32(v[n1 + n2..].to_vec()),
            )
        }
        WeightData::Q8_0Raw(b) => {
            let b1 = n1 / 32 * 34;
            let b2 = n2 / 32 * 34;
            let b3 = n3 / 32 * 34;
            assert_eq!(b.len(), b1 + b2 + b3);
            (
                WeightData::Q8_0Raw(b[0..b1].to_vec()),
                WeightData::Q8_0Raw(b[b1..b1 + b2].to_vec()),
                WeightData::Q8_0Raw(b[b1 + b2..].to_vec()),
            )
        }
        WeightData::Q4_0Raw(b) => {
            let b1 = n1 / 32 * 18;
            let b2 = n2 / 32 * 18;
            let b3 = n3 / 32 * 18;
            assert_eq!(b.len(), b1 + b2 + b3);
            (
                WeightData::Q4_0Raw(b[0..b1].to_vec()),
                WeightData::Q4_0Raw(b[b1..b1 + b2].to_vec()),
                WeightData::Q4_0Raw(b[b1 + b2..].to_vec()),
            )
        }
        WeightData::Q4_KRaw(b) => {
            assert!(n1.is_multiple_of(256) && n2.is_multiple_of(256) && n3.is_multiple_of(256),
                "Q4_K three-way split requires each part to be a multiple of 256; got {n1}, {n2}, {n3}");
            let b1 = n1 / 256 * 144;
            let b2 = n2 / 256 * 144;
            let b3 = n3 / 256 * 144;
            assert_eq!(b.len(), b1 + b2 + b3);
            (
                WeightData::Q4_KRaw(b[0..b1].to_vec()),
                WeightData::Q4_KRaw(b[b1..b1 + b2].to_vec()),
                WeightData::Q4_KRaw(b[b1 + b2..].to_vec()),
            )
        }
    }
}

fn split_weight_two(w: &WeightData, n1: usize, n2: usize) -> (WeightData, WeightData) {
    match w {
        WeightData::F32(v) => {
            assert_eq!(v.len(), n1 + n2);
            (
                WeightData::F32(v[0..n1].to_vec()),
                WeightData::F32(v[n1..].to_vec()),
            )
        }
        WeightData::Q8_0Raw(b) => {
            let b1 = n1 / 32 * 34;
            let b2 = n2 / 32 * 34;
            assert_eq!(b.len(), b1 + b2);
            (
                WeightData::Q8_0Raw(b[0..b1].to_vec()),
                WeightData::Q8_0Raw(b[b1..].to_vec()),
            )
        }
        WeightData::Q4_0Raw(b) => {
            let b1 = n1 / 32 * 18;
            let b2 = n2 / 32 * 18;
            assert_eq!(b.len(), b1 + b2);
            (
                WeightData::Q4_0Raw(b[0..b1].to_vec()),
                WeightData::Q4_0Raw(b[b1..].to_vec()),
            )
        }
        WeightData::Q4_KRaw(b) => {
            assert!(n1.is_multiple_of(256) && n2.is_multiple_of(256),
                "Q4_K two-way split requires each part to be a multiple of 256; got {n1}, {n2}");
            let b1 = n1 / 256 * 144;
            let b2 = n2 / 256 * 144;
            assert_eq!(b.len(), b1 + b2);
            (
                WeightData::Q4_KRaw(b[0..b1].to_vec()),
                WeightData::Q4_KRaw(b[b1..].to_vec()),
            )
        }
    }
}

/// Load a single tensor from a GGUF file, dequantizing to f32.
pub fn load_tensor<R: Read + Seek>(
    reader: &mut R,
    gguf: &GGUFFile,
    tensor_info: &GGUFTensorInfo,
) -> Result<Vec<f32>, WeightLoadError> {
    let data_offset = gguf.tensor_data_offset + tensor_info.offset;
    let data_size = tensor_info.data_size() as usize;
    let numel = tensor_info.numel() as usize;

    // Seek to tensor data
    reader.seek(SeekFrom::Start(data_offset))?;

    // Read raw bytes
    let mut raw = vec![0u8; data_size];
    reader.read_exact(&mut raw)?;

    // Dequantize to f32
    dequantize(&raw, tensor_info.ggml_type, numel)
}

/// Load a tensor by name from a GGUF file.
pub fn load_tensor_by_name<R: Read + Seek>(
    reader: &mut R,
    gguf: &GGUFFile,
    name: &str,
) -> Result<Vec<f32>, WeightLoadError> {
    let tensor_info = gguf
        .tensor(name)
        .ok_or_else(|| WeightLoadError::TensorNotFound(name.to_string()))?;
    load_tensor(reader, gguf, tensor_info)
}

/// Dequantize Q8_0 raw bytes to f32 — public for use by export-weights.
///
/// `raw`: raw Q8_0 bytes (34 bytes/block: 2-byte f16 scale + 32 int8 values)
/// `numel`: total number of elements
/// Dequantize raw Q4_K bytes (144 bytes per 256-element super-block) back
/// to f32.  Used by the export-weights path when an embedding tensor is
/// stored as Q4_K in memory but needs to be written as flat f32 in the
/// final weight file.
pub fn dequantize_q4_k_to_f32(raw: &[u8], numel: usize) -> Vec<f32> {
    dequant_q4_k(raw, numel)
}

pub fn dequantize_q8_0_to_f32(raw: &[u8], numel: usize) -> Vec<f32> {
    dequant_q8_0(raw, numel)
}

/// Dequantize raw bytes to f32 based on GGML type.
fn dequantize(data: &[u8], ggml_type: GGMLType, numel: usize) -> Result<Vec<f32>, WeightLoadError> {
    match ggml_type {
        GGMLType::F32 => Ok(dequant_f32(data, numel)),
        GGMLType::F16 => Ok(dequant_f16(data, numel)),
        GGMLType::BF16 => Ok(dequant_bf16(data, numel)),
        GGMLType::Q8_0 => Ok(dequant_q8_0(data, numel)),
        GGMLType::Q4_0 => Ok(dequant_q4_0(data, numel)),
        GGMLType::Q4_1 => Ok(dequant_q4_1(data, numel)),
        GGMLType::Q6K => Ok(dequant_q6_k(data, numel)),
        GGMLType::Q5K => Ok(dequant_q5_k(data, numel)),
        GGMLType::Q4K => Ok(dequant_q4_k(data, numel)),
        GGMLType::Q8K => Ok(dequant_q8_k(data, numel)),
        GGMLType::Q3K => Ok(dequant_q3_k(data, numel)),
        GGMLType::Q2K => Ok(dequant_q2_k(data, numel)),
        other => Err(WeightLoadError::UnsupportedType(other)),
    }
}

/// F32: just reinterpret bytes.
fn dequant_f32(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    for (i, chunk) in data.chunks_exact(4).enumerate().take(numel) {
        output[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    output
}

/// F16 → F32 conversion.
fn dequant_f16(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    for (i, chunk) in data.chunks_exact(2).enumerate().take(numel) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        output[i] = f16_to_f32(bits);
    }
    output
}

/// BF16 → F32 conversion.
fn dequant_bf16(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    for (i, chunk) in data.chunks_exact(2).enumerate().take(numel) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        // BF16 is just the upper 16 bits of F32
        output[i] = f32::from_bits((bits as u32) << 16);
    }
    output
}

/// Q8_0 dequantization.
/// Block layout: 2 bytes scale (f16) + 32 bytes quantized (int8).
/// Block size: 32 elements.
fn dequant_q8_0(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    let block_size = 32;
    let type_size = 34; // 2 (scale) + 32 (data)
    let num_blocks = numel.div_ceil(block_size);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * type_size;
        if block_start + type_size > data.len() {
            break;
        }

        // Scale is f16
        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = f16_to_f32(scale_bits);

        // Quantized values are int8
        for j in 0..block_size {
            let out_idx = block_idx * block_size + j;
            if out_idx >= numel {
                break;
            }
            let quant = data[block_start + 2 + j] as i8;
            output[out_idx] = quant as f32 * scale;
        }
    }

    output
}

/// Q4_0 dequantization.
/// Block layout: 2 bytes scale (f16) + 16 bytes quantized (4-bit pairs).
/// Block size: 32 elements (packed as 16 bytes, 2 elements per byte).
fn dequant_q4_0(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    let block_size = 32;
    let type_size = 18; // 2 (scale) + 16 (data)
    let num_blocks = numel.div_ceil(block_size);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * type_size;
        if block_start + type_size > data.len() {
            break;
        }

        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = f16_to_f32(scale_bits);

        for j in 0..16 {
            let byte = data[block_start + 2 + j];
            let lo = (byte & 0x0F) as i32 - 8; // 4-bit unsigned → signed (offset by 8)
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;

            let out_idx_lo = block_idx * block_size + j;
            let out_idx_hi = block_idx * block_size + j + 16;

            if out_idx_lo < numel {
                output[out_idx_lo] = lo as f32 * scale;
            }
            if out_idx_hi < numel {
                output[out_idx_hi] = hi as f32 * scale;
            }
        }
    }

    output
}

/// Q4_1 dequantization.
/// Block layout: 2 bytes scale (f16) + 2 bytes min (f16) + 16 bytes quantized.
/// Block size: 32 elements.
fn dequant_q4_1(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    let block_size = 32;
    let type_size = 20; // 2 (scale) + 2 (min) + 16 (data)
    let num_blocks = numel.div_ceil(block_size);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * type_size;
        if block_start + type_size > data.len() {
            break;
        }

        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let min_bits = u16::from_le_bytes([data[block_start + 2], data[block_start + 3]]);
        let scale = f16_to_f32(scale_bits);
        let min = f16_to_f32(min_bits);

        for j in 0..16 {
            let byte = data[block_start + 4 + j];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;

            let out_idx_lo = block_idx * block_size + j;
            let out_idx_hi = block_idx * block_size + j + 16;

            if out_idx_lo < numel {
                output[out_idx_lo] = lo * scale + min;
            }
            if out_idx_hi < numel {
                output[out_idx_hi] = hi * scale + min;
            }
        }
    }

    output
}

/// Unpack 6-bit scale and min values from the 12-byte packed K-quant scale format.
///
/// The 12 bytes encode 8 scales and 8 mins, each 6 bits wide:
/// - Bytes 0..3: low 6 bits of scales[0..3]
/// - Bytes 4..7: low 6 bits of mins[0..3]
/// - Bytes 8..11: high 2 bits of scales[4..7] combined with low 4 bits, and
///   high 2 bits of mins[4..7] combined with low 4 bits
///
/// Matches GGML `get_scale_min_k4`.
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let sc = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (sc, m)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Q6_K dequantization.
/// Block layout (210 bytes per 256 elements):
///   - 128 bytes ql: lower 4 bits of 6-bit quantized values
///   - 64 bytes qh: upper 2 bits of 6-bit quantized values
///   - 16 bytes scales: per-16-element int8 scales
///   - 2 bytes d: super-block scale (f16)
///
/// Matches GGML `dequantize_row_q6_K`.
fn dequant_q6_k(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    let block_size = 256;
    let type_size = 210;
    let num_blocks = numel.div_ceil(block_size);

    for block_idx in 0..num_blocks {
        let bs = block_idx * type_size;
        if bs + type_size > data.len() {
            break;
        }

        let ql = &data[bs..bs + 128];
        let qh = &data[bs + 128..bs + 192];
        let scales = &data[bs + 192..bs + 208];
        let d_bits = u16::from_le_bytes([data[bs + 208], data[bs + 209]]);
        let d = f16_to_f32(d_bits);

        let out_base = block_idx * block_size;

        // Process in two 128-element chunks, matching GGML reference.
        // Each chunk processes 32 iterations, producing 4 output values each.
        for n_off in (0..block_size).step_by(128) {
            let ql_off = n_off / 2; // ql advances by 64 per 128-element chunk
            let qh_off = n_off / 4; // qh advances by 32 per 128-element chunk
            let sc_off = n_off / 16; // scales advance by 8 per 128-element chunk

            for l in 0..32 {
                let is = l / 16; // sub-group index within this chunk (0 or 1)

                let q1 = ((ql[ql_off + l] & 0x0F) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
                let q2 =
                    ((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

                let sc1 = scales[sc_off + is] as i8;
                let sc2 = scales[sc_off + is + 2] as i8;
                let sc3 = scales[sc_off + is + 4] as i8;
                let sc4 = scales[sc_off + is + 6] as i8;

                let out_idx = out_base + n_off + l;
                if out_idx < numel {
                    output[out_idx] = d * sc1 as f32 * q1 as f32;
                }
                if out_idx + 32 < numel {
                    output[out_idx + 32] = d * sc2 as f32 * q2 as f32;
                }
                if out_idx + 64 < numel {
                    output[out_idx + 64] = d * sc3 as f32 * q3 as f32;
                }
                if out_idx + 96 < numel {
                    output[out_idx + 96] = d * sc4 as f32 * q4 as f32;
                }
            }
        }
    }

    output
}

/// Q5_K dequantization.
/// Block layout (176 bytes per 256 elements):
///   - 2 bytes d: super-block scale (f16)
///   - 2 bytes dmin: super-block min (f16)
///   - 12 bytes scales: 6-bit packed scales and mins for 8 sub-blocks
///   - 32 bytes qh: high bit (5th bit) for each of 256 elements
///   - 128 bytes qs: low 4 bits for each of 256 elements
///
/// Dequantization: value = d * scale_j * q5 - dmin * min_j
/// where q5 is the 5-bit unsigned value (4 low bits from qs + 1 high bit from qh).
///
/// Matches GGML `dequantize_row_q5_K`.
fn dequant_q5_k(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    let block_size = 256;
    let type_size = 176;
    let num_blocks = numel.div_ceil(block_size);

    for block_idx in 0..num_blocks {
        let bs = block_idx * type_size;
        if bs + type_size > data.len() {
            break;
        }

        let d_bits = u16::from_le_bytes([data[bs], data[bs + 1]]);
        let dmin_bits = u16::from_le_bytes([data[bs + 2], data[bs + 3]]);
        let d = f16_to_f32(d_bits);
        let dmin = f16_to_f32(dmin_bits);

        let scales = &data[bs + 4..bs + 16];
        let qh = &data[bs + 16..bs + 48];
        let qs = &data[bs + 48..bs + 176];

        let out_base = block_idx * block_size;

        // Process in 64-element chunks matching GGML reference.
        // Each chunk uses two sub-blocks of 32 elements:
        //   first 32: low nibble of qs[l], qh bit at mask u1
        //   next 32:  high nibble of qs[l], qh bit at mask u2
        let mut ql_off = 0usize;
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for chunk in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            for l in 0..32 {
                let out_idx = out_base + chunk * 64 + l;
                if out_idx >= numel {
                    break;
                }
                let q_lo = (qs[ql_off + l] & 0x0F) as u32;
                let qh_bit = if qh[l] & u1 != 0 { 16u32 } else { 0u32 };
                output[out_idx] = d1 * (q_lo + qh_bit) as f32 - m1;
            }
            for l in 0..32 {
                let out_idx = out_base + chunk * 64 + 32 + l;
                if out_idx >= numel {
                    break;
                }
                let q_hi = (qs[ql_off + l] >> 4) as u32;
                let qh_bit = if qh[l] & u2 != 0 { 16u32 } else { 0u32 };
                output[out_idx] = d2 * (q_hi + qh_bit) as f32 - m2;
            }
            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    output
}

/// Q4_K dequantization.
/// Block layout (144 bytes per 256 elements):
///   - 2 bytes d: super-block scale (f16)
///   - 2 bytes dmin: super-block min (f16)
///   - 12 bytes scales: 6-bit packed scales and mins for 8 sub-blocks
///   - 128 bytes qs: 4-bit quantized values (256 elements, 2 per byte)
///
/// Matches GGML `dequantize_row_q4_K`.
fn dequant_q4_k(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    let block_size = 256;
    let type_size = 144;
    let num_blocks = numel.div_ceil(block_size);

    for block_idx in 0..num_blocks {
        let bs = block_idx * type_size;
        if bs + type_size > data.len() {
            break;
        }

        let d_bits = u16::from_le_bytes([data[bs], data[bs + 1]]);
        let dmin_bits = u16::from_le_bytes([data[bs + 2], data[bs + 3]]);
        let d = f16_to_f32(d_bits);
        let dmin = f16_to_f32(dmin_bits);

        let scales = &data[bs + 4..bs + 16];
        let qs = &data[bs + 16..bs + 144];

        let out_base = block_idx * block_size;

        // Process in 64-element chunks matching GGML reference.
        // Each chunk has two 32-element sub-blocks:
        //   first 32: low nibble of qs[l]
        //   next 32:  high nibble of qs[l]
        let mut q_off = 0usize;
        let mut is = 0usize;
        for chunk in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            for l in 0..32 {
                let out_idx = out_base + chunk * 64 + l;
                if out_idx >= numel {
                    break;
                }
                output[out_idx] = d1 * (qs[q_off + l] & 0x0F) as f32 - m1;
            }
            for l in 0..32 {
                let out_idx = out_base + chunk * 64 + 32 + l;
                if out_idx >= numel {
                    break;
                }
                output[out_idx] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            }
            q_off += 32;
            is += 2;
        }
    }

    output
}

/// Q8_K dequantization.
/// Block layout (292 bytes per 256 elements): f32 scale + 256 int8 values.
fn dequant_q8_k(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    let block_size = 256;
    let type_size = 292;
    let num_blocks = numel.div_ceil(block_size);

    for block_idx in 0..num_blocks {
        let bs = block_idx * type_size;
        if bs + type_size > data.len() {
            break;
        }

        let scale = f32::from_le_bytes([data[bs], data[bs + 1], data[bs + 2], data[bs + 3]]);

        for j in 0..block_size {
            if block_idx * block_size + j >= numel {
                break;
            }
            let q = data[bs + 4 + j] as i8;
            output[block_idx * block_size + j] = scale * q as f32;
        }
    }

    output
}

/// Q3_K dequantization.
/// Block layout (110 bytes per 256 elements).
fn dequant_q3_k(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    let block_size = 256;
    let type_size = 110;
    let num_blocks = numel.div_ceil(block_size);

    for block_idx in 0..num_blocks {
        let bs = block_idx * type_size;
        if bs + type_size > data.len() {
            break;
        }

        // Simplified dequantization
        let hmask = &data[bs..bs + 32];
        let qs = &data[bs + 32..bs + 96];
        let scales_raw = &data[bs + 96..bs + 108];
        let d_bits = u16::from_le_bytes([data[bs + 108], data[bs + 109]]);
        let d = f16_to_f32(d_bits);

        for j in 0..block_size {
            if block_idx * block_size + j >= numel {
                break;
            }
            let group = j / 16;
            let sc = if group < scales_raw.len() {
                ((scales_raw[group / 2] >> ((group % 2) * 4)) & 0x0F) as i32 - 8
            } else {
                0
            };

            let byte_idx = j * 3 / 8;
            let bit_offset = (j * 3) % 8;
            let q3 = if byte_idx < qs.len() {
                ((qs[byte_idx] >> bit_offset) & 0x07) as i32 - 4
            } else {
                0
            };
            let hbit = if j / 8 < hmask.len() {
                ((hmask[j / 8] >> (j % 8)) & 1) as i32
            } else {
                0
            };
            let q = q3 - hbit * 4;
            output[block_idx * block_size + j] = d * sc as f32 * q as f32;
        }
    }

    output
}

/// Q2_K dequantization.
/// Block layout (84 bytes per 256 elements).
fn dequant_q2_k(data: &[u8], numel: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; numel];
    let block_size = 256;
    let type_size = 84;
    let num_blocks = numel.div_ceil(block_size);

    for block_idx in 0..num_blocks {
        let bs = block_idx * type_size;
        if bs + type_size > data.len() {
            break;
        }

        let scales = &data[bs..bs + 16];
        let qs = &data[bs + 16..bs + 80];
        let d_bits = u16::from_le_bytes([data[bs + 80], data[bs + 81]]);
        let dmin_bits = u16::from_le_bytes([data[bs + 82], data[bs + 83]]);
        let d = f16_to_f32(d_bits);
        let dmin = f16_to_f32(dmin_bits);

        for j in 0..block_size {
            if block_idx * block_size + j >= numel {
                break;
            }
            let group = j / 16;
            let sc = scales[group] & 0x0F;
            let m = (scales[group] >> 4) & 0x0F;

            let byte_idx = j / 4;
            let q = if byte_idx < qs.len() {
                (qs[byte_idx] >> ((j % 4) * 2)) & 0x03
            } else {
                0
            };
            output[block_idx * block_size + j] = d * sc as f32 * q as f32 - dmin * m as f32;
        }
    }

    output
}

/// Convert IEEE 754 half-precision float (f16) to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero
            return f32::from_bits(sign << 31);
        }
        // Subnormal: convert to normalized f32
        let mut m = mantissa;
        let mut e: i32 = -14; // f16 subnormal exponent bias
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF; // remove implicit leading 1
        let f32_exp = ((e + 127) as u32) & 0xFF;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }

    if exponent == 31 {
        // Inf or NaN
        let f32_mantissa = mantissa << 13;
        return f32::from_bits((sign << 31) | (0xFF << 23) | f32_mantissa);
    }

    // Normal number
    let f32_exp = (exponent as i32 - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

/// F32 → F16 conversion (round to nearest, ties to even).
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 0xFF {
        // Inf / NaN
        let h_mant = (mant >> 13) & 0x3FF;
        return ((sign << 15) | (0x1F << 10) | h_mant) as u16;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow → f16 infinity
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    if unbiased < -24 {
        // Underflow → zero
        return (sign << 15) as u16;
    }
    if unbiased < -14 {
        // Subnormal f16
        let shift = (-14 - unbiased) as u32;
        let h_mant = (mant | 0x80_0000) >> (14 + shift);
        return ((sign << 15) | h_mant) as u16;
    }

    let h_exp = (unbiased + 15) as u32;
    let h_mant = mant >> 13;
    ((sign << 15) | (h_exp << 10) | h_mant) as u16
}

/// Quantize an f32 slice to Q8_0 raw bytes.
///
/// Q8_0 block layout (32 elements → 34 bytes):
/// - 2 bytes: f16 scale (absmax / 127)
/// - 32 bytes: int8 quantized values
///
/// Used to repack K-quant tensors (Q4_K, Q5_K, Q6_K, ...) into the Q8_0
/// format that ForgeLLM's CPU kernels consume natively. Trades ~2× weight
/// memory vs. native K-quant for full kernel compatibility — native K-quant
/// kernels are a separate (future) work item.
pub fn quantize_f32_to_q8_0(data: &[f32]) -> Vec<u8> {
    let num_blocks = data.len().div_ceil(32);
    let mut out = vec![0u8; num_blocks * 34];

    for blk in 0..num_blocks {
        let base = blk * 32;
        let end = (base + 32).min(data.len());
        let block = &data[base..end];

        let mut amax = 0.0f32;
        for &v in block {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        let ob = blk * 34;
        let scale_bits = f32_to_f16(scale);
        out[ob] = scale_bits as u8;
        out[ob + 1] = (scale_bits >> 8) as u8;

        for (j, &v) in block.iter().enumerate() {
            let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            out[ob + 2 + j] = q as u8;
        }
        // Tail elements (when data.len() is not a multiple of 32) are zero-filled.
    }

    out
}

/// Quantize an f32 slice to Q4_0 raw bytes.
///
/// Q4_0 block layout (32 elements → 18 bytes):
/// - 2 bytes: f16 scale (max absolute value / 8)
/// - 16 bytes: packed 4-bit unsigned values (offset by 8).
///   Byte j contains element j in low nibble and element j+16 in high nibble.
pub fn quantize_f32_to_q4_0(data: &[f32]) -> Vec<u8> {
    let num_blocks = data.len().div_ceil(32);
    let mut out = vec![0u8; num_blocks * 18];

    for blk in 0..num_blocks {
        let base = blk * 32;
        let end = (base + 32).min(data.len());
        let block_data = &data[base..end];

        // Find max absolute value
        let mut amax = 0.0f32;
        for &v in block_data {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }

        // Scale: maps [-8*scale, 7*scale] to the value range
        let scale = amax / 8.0;
        let inv_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        let ob = blk * 18;
        let scale_bits = f32_to_f16(scale);
        out[ob] = scale_bits as u8;
        out[ob + 1] = (scale_bits >> 8) as u8;

        // Quantize: value → round(value / scale) + 8, clamp to [0, 15]
        for j in 0..16 {
            let lo_idx = j;
            let hi_idx = j + 16;
            let lo_val = if base + lo_idx < data.len() {
                data[base + lo_idx]
            } else {
                0.0
            };
            let hi_val = if base + hi_idx < data.len() {
                data[base + hi_idx]
            } else {
                0.0
            };

            let lo_q = ((lo_val * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;
            let hi_q = ((hi_val * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;

            out[ob + 2 + j] = lo_q | (hi_q << 4);
        }
    }

    out
}

/// Quantize an f32 slice to Q4_K raw bytes.
///
/// Q4_K block layout (256 elements → 144 bytes):
///   [0..2]    d: f16 super-block scale
///   [2..4]    dmin: f16 super-block min
///   [4..16]   12 bytes packing 8 sub-block (6-bit scale, 6-bit min) pairs
///   [16..144] 128 bytes packing 256 × 4-bit unsigned weights
///
/// Algorithm: split each 256-element block into 8 sub-blocks of 32 elements.
/// Per sub-block j, compute `(min_j, max_j)` and derive
/// `sub_scale_j = (max_j - min_j) / 15` plus offset `min_j`.  The super-block
/// scales `d` and `dmin` are chosen so the per-sub-block 6-bit values fit:
///   d = max(sub_scale_j) / 63          (all sub_scale_j ≥ 0)
///   dmin = max(|min_j|) / 63, with the sign carried on dmin itself.
///
/// This is a simple affine quantizer — coarser than GGML's iterative
/// `make_qkx2_quants` but adequate for re-packing Q6_K / Q5_K / etc. that
/// have already been dequantized once.  Used by the loader when presenting
/// a mixed Q4_K_M GGUF to the Q4_K code path.
pub fn quantize_f32_to_q4_k(data: &[f32]) -> Vec<u8> {
    const BLOCK: usize = 256;
    const BYTES: usize = 144;
    const SUB: usize = 32;

    let num_sb = data.len().div_ceil(BLOCK);
    let mut out = vec![0u8; num_sb * BYTES];

    for sb in 0..num_sb {
        let base = sb * BLOCK;
        let end = (base + BLOCK).min(data.len());
        let block = &data[base..end];
        let mut padded = [0.0f32; BLOCK];
        padded[..block.len()].copy_from_slice(block);

        // Per sub-block: (min, scale, q4_values).
        let mut sub_scales_f = [0.0f32; 8];
        let mut sub_mins_f = [0.0f32; 8];
        let mut sub_q: [[u8; SUB]; 8] = [[0; SUB]; 8];
        for j in 0..8 {
            let s0 = j * SUB;
            let sub = &padded[s0..s0 + SUB];
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &v in sub {
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
            }
            let scale = (mx - mn) / 15.0;
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            sub_scales_f[j] = scale;
            sub_mins_f[j] = mn;
            for (i, &v) in sub.iter().enumerate() {
                let q = ((v - mn) * inv_scale).round().clamp(0.0, 15.0) as u8;
                sub_q[j][i] = q;
            }
        }

        // Super-block scales.  d carries sub_scale magnitudes; dmin carries
        // sub_min magnitudes (signed — keep the max-magnitude min's sign).
        let max_scale = sub_scales_f
            .iter()
            .fold(0.0f32, |a, &b| if b > a { b } else { a });
        let d = if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 };
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

        // For dmin, pick the min_j with largest magnitude, preserve sign.
        let mut max_mag_min = 0.0f32;
        let mut sign_ref = 0.0f32;
        for &m in &sub_mins_f {
            if m.abs() > max_mag_min {
                max_mag_min = m.abs();
                sign_ref = m;
            }
        }
        // Dequant: w[i] = d * sub_scale[j] * q[i] - dmin * sub_min[j]
        // We want -dmin * sub_min[j] = min_j  ⟹  dmin * sub_min[j] = -min_j
        // With sub_min ≥ 0, choose dmin = -sign(min_j-with-max-mag) * mag / 63.
        let dmin = if max_mag_min > 0.0 {
            -sign_ref.signum() * (max_mag_min / 63.0)
        } else {
            0.0
        };
        let inv_dmin = if dmin.abs() > 0.0 { 1.0 / dmin } else { 0.0 };

        // 6-bit per-sub-block scale and min.
        let mut sub_scale_q = [0u8; 8];
        let mut sub_min_q = [0u8; 8];
        for j in 0..8 {
            sub_scale_q[j] = (sub_scales_f[j] * inv_d).round().clamp(0.0, 63.0) as u8;
            // sub_min[j] * dmin = -min_j  ⟹  sub_min[j] = -min_j / dmin
            let smq = (-sub_mins_f[j] * inv_dmin).round().clamp(0.0, 63.0) as u8;
            sub_min_q[j] = smq;
        }

        // Write super-block header.
        let ob = sb * BYTES;
        let d_bits = f32_to_f16(d);
        let dmin_bits = f32_to_f16(dmin);
        out[ob] = d_bits as u8;
        out[ob + 1] = (d_bits >> 8) as u8;
        out[ob + 2] = dmin_bits as u8;
        out[ob + 3] = (dmin_bits >> 8) as u8;

        // Pack scales/mins — inverse of get_scale_min_k4.
        for i in 0..4 {
            out[ob + 4 + i] = (sub_scale_q[i] & 63) | ((sub_scale_q[i + 4] >> 4) & 0x3) << 6;
            out[ob + 8 + i] = (sub_min_q[i] & 63) | ((sub_min_q[i + 4] >> 4) & 0x3) << 6;
            out[ob + 12 + i] = (sub_scale_q[i + 4] & 0x0F) | ((sub_min_q[i + 4] & 0x0F) << 4);
        }

        // Pack qs: 128 bytes, 4 chunks of 32 bytes.  Each byte holds two
        // nibbles: the low nibble belongs to sub-block 2*chunk, the high
        // nibble to sub-block 2*chunk+1.
        for c in 0..4 {
            for i in 0..32 {
                let lo = sub_q[2 * c][i] & 0x0F;
                let hi = sub_q[2 * c + 1][i] & 0x0F;
                out[ob + 16 + c * 32 + i] = lo | (hi << 4);
            }
        }
    }

    out
}

/// Scalar dot product of a Q4_K-quantized weight row and a Q8_0-quantized
/// input row, both of length `k`.
///
/// `k` must be a multiple of 256 (Q4_K super-block size).  A single super-
/// block of Q4_K weights (144 bytes) pairs with eight Q8_0 input blocks
/// (8 × 34 = 272 bytes), each covering 32 of the super-block's 256 elements.
///
/// Reference implementation — no SIMD.  Used by unit tests and as the
/// baseline that the generated-code Q4_K kernel must match within a small
/// floating-point tolerance.
///
/// Layout:
/// - `a_q4k`: [f16 d, f16 dmin, 12B scales, 128B qs] × num_superblocks.
/// - `a_q8`: [f16 scale, 32B i8] × num_q8_blocks.
///
/// The dequant formula (per 32-element sub-block j):
///     w[i] = d * sub_scale_j * q[i] - dmin * sub_min_j
/// where `q[i]` is the 4-bit unsigned weight value and (sub_scale_j,
/// sub_min_j) are 6-bit values unpacked from the 12-byte scales blob via
/// `get_scale_min_k4`.
pub fn dot_q4_k_q8_0(weight_q4k: &[u8], input_q8: &[u8], k: usize) -> f32 {
    const Q4K_BLOCK: usize = 256;
    const Q4K_BYTES: usize = 144;
    const Q8_BYTES: usize = 34;
    assert!(
        k.is_multiple_of(Q4K_BLOCK),
        "Q4_K dot requires k multiple of 256, got {k}"
    );

    let num_sb = k / Q4K_BLOCK;
    let mut acc = 0.0f32;

    for sb in 0..num_sb {
        let wb = sb * Q4K_BYTES;
        let d_bits = u16::from_le_bytes([weight_q4k[wb], weight_q4k[wb + 1]]);
        let dmin_bits = u16::from_le_bytes([weight_q4k[wb + 2], weight_q4k[wb + 3]]);
        let d = f16_to_f32(d_bits);
        let dmin = f16_to_f32(dmin_bits);
        let scales = &weight_q4k[wb + 4..wb + 16];
        let qs = &weight_q4k[wb + 16..wb + 144];

        // Eight 32-element Q8_0 input blocks back the 256 Q4_K weights.
        let x_block_base = sb * 8 * Q8_BYTES;

        // Process four 64-element chunks (two sub-blocks per chunk), matching
        // the GGML reference.  chunk*64 .. chunk*64+32 uses low nibbles
        // (sub-block 2*chunk); chunk*64+32 .. chunk*64+64 uses high nibbles
        // (sub-block 2*chunk+1).  Both nibble pairs share the same 32-byte
        // qs slice `qs[chunk*32 .. chunk*32+32]`.
        for chunk in 0..4 {
            let is = chunk * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1_w = d * sc1 as f32;
            let m1_w = dmin * m1 as f32;
            let d2_w = d * sc2 as f32;
            let m2_w = dmin * m2 as f32;

            let qs_chunk = &qs[chunk * 32..(chunk + 1) * 32];

            // Low-nibble sub-block.
            let x1_off = x_block_base + is * Q8_BYTES;
            let x1_scale = f16_to_f32(u16::from_le_bytes([
                input_q8[x1_off],
                input_q8[x1_off + 1],
            ]));
            let x1_i8 = &input_q8[x1_off + 2..x1_off + 34];
            let mut dot1 = 0i32;
            let mut sum1 = 0i32;
            for i in 0..32 {
                let q4 = (qs_chunk[i] & 0x0F) as i32;
                let xi = x1_i8[i] as i8 as i32;
                dot1 += xi * q4;
                sum1 += xi;
            }
            acc += x1_scale * (d1_w * dot1 as f32 - m1_w * sum1 as f32);

            // High-nibble sub-block.
            let x2_off = x_block_base + (is + 1) * Q8_BYTES;
            let x2_scale = f16_to_f32(u16::from_le_bytes([
                input_q8[x2_off],
                input_q8[x2_off + 1],
            ]));
            let x2_i8 = &input_q8[x2_off + 2..x2_off + 34];
            let mut dot2 = 0i32;
            let mut sum2 = 0i32;
            for i in 0..32 {
                let q4 = (qs_chunk[i] >> 4) as i32;
                let xi = x2_i8[i] as i8 as i32;
                dot2 += xi * q4;
                sum2 += xi;
            }
            acc += x2_scale * (d2_w * dot2 as f32 - m2_w * sum2 as f32);
        }
    }

    acc
}

/// AArch64 NEON variant of `dot_q4_k_q8_0`.  Inline `sdot` for the int8×4-bit
/// dot product; per-sub-block `vpaddlq_s8 + vaddvq_s16` for the
/// `Σ x_q8` reduction needed to apply the affine `dmin` correction.
#[cfg(target_arch = "aarch64")]
pub fn dot_q4_k_q8_0_neon(weight_q4k: &[u8], input_q8: &[u8], k: usize) -> f32 {
    use std::arch::aarch64::*;
    const Q4K_BLOCK: usize = 256;
    const Q4K_BYTES: usize = 144;
    const Q8_BYTES: usize = 34;
    assert!(
        k.is_multiple_of(Q4K_BLOCK),
        "Q4_K NEON dot requires k multiple of 256, got {k}"
    );

    let num_sb = k / Q4K_BLOCK;
    let mut acc = 0.0f32;

    for sb in 0..num_sb {
        let wb = sb * Q4K_BYTES;
        let d_bits = u16::from_le_bytes([weight_q4k[wb], weight_q4k[wb + 1]]);
        let dmin_bits = u16::from_le_bytes([weight_q4k[wb + 2], weight_q4k[wb + 3]]);
        let d = f16_to_f32(d_bits);
        let dmin = f16_to_f32(dmin_bits);
        let scales = &weight_q4k[wb + 4..wb + 16];
        let qs = &weight_q4k[wb + 16..wb + 144];
        let x_block_base = sb * 8 * Q8_BYTES;

        for chunk in 0..4 {
            let is = chunk * 2;
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1_w = d * sc1 as f32;
            let m1_w = dmin * m1 as f32;
            let d2_w = d * sc2 as f32;
            let m2_w = dmin * m2 as f32;

            unsafe {
                let mask = vdupq_n_u8(0x0F);
                let qs_ptr = qs[chunk * 32..].as_ptr();
                let qs_v0: uint8x16_t = vld1q_u8(qs_ptr);
                let qs_v1: uint8x16_t = vld1q_u8(qs_ptr.add(16));

                // Sub-block 1: low nibbles (0..15).
                let q1_lo = vreinterpretq_s8_u8(vandq_u8(qs_v0, mask));
                let q1_hi = vreinterpretq_s8_u8(vandq_u8(qs_v1, mask));
                // Sub-block 2: high nibbles.
                let q2_lo = vreinterpretq_s8_u8(vshrq_n_u8::<4>(qs_v0));
                let q2_hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(qs_v1));

                // Sub-block 1 input.
                let x1_off = x_block_base + is * Q8_BYTES;
                let x1_scale = f16_to_f32(u16::from_le_bytes([
                    input_q8[x1_off],
                    input_q8[x1_off + 1],
                ]));
                let x1_ptr = input_q8[x1_off + 2..].as_ptr() as *const i8;
                let x1_lo: int8x16_t = vld1q_s8(x1_ptr);
                let x1_hi: int8x16_t = vld1q_s8(x1_ptr.add(16));

                let mut dot1_v = vdupq_n_s32(0);
                core::arch::asm!(
                    "sdot {acc:v}.4s, {w0:v}.16b, {x0:v}.16b",
                    "sdot {acc:v}.4s, {w1:v}.16b, {x1:v}.16b",
                    acc = inout(vreg) dot1_v,
                    w0 = in(vreg) q1_lo,
                    x0 = in(vreg) x1_lo,
                    w1 = in(vreg) q1_hi,
                    x1 = in(vreg) x1_hi,
                    options(nostack),
                );
                let dot1 = vaddvq_s32(dot1_v) as i32;
                let sum1_v = vaddq_s16(vpaddlq_s8(x1_lo), vpaddlq_s8(x1_hi));
                let sum1 = vaddvq_s16(sum1_v) as i32;
                acc += x1_scale * (d1_w * dot1 as f32 - m1_w * sum1 as f32);

                // Sub-block 2 input.
                let x2_off = x_block_base + (is + 1) * Q8_BYTES;
                let x2_scale = f16_to_f32(u16::from_le_bytes([
                    input_q8[x2_off],
                    input_q8[x2_off + 1],
                ]));
                let x2_ptr = input_q8[x2_off + 2..].as_ptr() as *const i8;
                let x2_lo: int8x16_t = vld1q_s8(x2_ptr);
                let x2_hi: int8x16_t = vld1q_s8(x2_ptr.add(16));

                let mut dot2_v = vdupq_n_s32(0);
                core::arch::asm!(
                    "sdot {acc:v}.4s, {w0:v}.16b, {x0:v}.16b",
                    "sdot {acc:v}.4s, {w1:v}.16b, {x1:v}.16b",
                    acc = inout(vreg) dot2_v,
                    w0 = in(vreg) q2_lo,
                    x0 = in(vreg) x2_lo,
                    w1 = in(vreg) q2_hi,
                    x1 = in(vreg) x2_hi,
                    options(nostack),
                );
                let dot2 = vaddvq_s32(dot2_v) as i32;
                let sum2_v = vaddq_s16(vpaddlq_s8(x2_lo), vpaddlq_s8(x2_hi));
                let sum2 = vaddvq_s16(sum2_v) as i32;
                acc += x2_scale * (d2_w * dot2 as f32 - m2_w * sum2 as f32);
            }
        }
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_basic() {
        // 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
        // 0.5 in f16 = 0x3800
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
        // 2.0 in f16 = 0x4000
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn f16_special_values() {
        // +inf = 0x7C00
        assert!(f16_to_f32(0x7C00).is_infinite());
        // NaN = 0x7E00
        assert!(f16_to_f32(0x7E00).is_nan());
        // -0.0 = 0x8000
        assert_eq!(f16_to_f32(0x8000), -0.0);
    }

    /// Build a Q4_K super-block (144 bytes) with user-chosen parameters.
    /// Returns a vec of exactly 144 bytes.
    /// - `d`, `dmin`: super-block scales (f16-converted internally).
    /// - `sub_scales[i]`, `sub_mins[i]`: 6-bit scale/min for sub-block i (0..8).
    /// - `qs_low_bytes[c][i]`, `qs_high_bytes[c][i]`: low/high nibbles for
    ///   chunk `c` (0..4), element `i` (0..32).  Only the low 4 bits matter.
    fn build_q4k_superblock(
        d: f32,
        dmin: f32,
        sub_scales: [u8; 8],
        sub_mins: [u8; 8],
        qs_lo: &[[u8; 32]; 4],
        qs_hi: &[[u8; 32]; 4],
    ) -> Vec<u8> {
        let mut out = vec![0u8; 144];
        let d_bits = f32_to_f16(d);
        let dmin_bits = f32_to_f16(dmin);
        out[0] = d_bits as u8;
        out[1] = (d_bits >> 8) as u8;
        out[2] = dmin_bits as u8;
        out[3] = (dmin_bits >> 8) as u8;
        // Scales layout — inverse of get_scale_min_k4:
        //   scales[0..3] = low 6 bits of sub_scale[0..3]  ∪ top 2 bits of sub_scale[4..7]
        //   scales[4..7] = low 6 bits of sub_min[0..3]    ∪ top 2 bits of sub_min[4..7]
        //   scales[8..11] = (sub_scale[4..7] & 0x0F) | ((sub_min[4..7] & 0x0F) << 4)
        for i in 0..4 {
            let sc_lo = sub_scales[i] & 63;
            let sc_hi_top = (sub_scales[i + 4] >> 4) & 0x3;
            out[4 + i] = sc_lo | (sc_hi_top << 6);
        }
        for i in 0..4 {
            let m_lo = sub_mins[i] & 63;
            let m_hi_top = (sub_mins[i + 4] >> 4) & 0x3;
            out[8 + i] = m_lo | (m_hi_top << 6);
        }
        for i in 0..4 {
            let sc_lo4 = sub_scales[i + 4] & 0x0F;
            let m_lo4 = sub_mins[i + 4] & 0x0F;
            out[12 + i] = sc_lo4 | (m_lo4 << 4);
        }
        // qs: 128 bytes, 32 bytes per chunk.
        for c in 0..4 {
            for i in 0..32 {
                let lo = qs_lo[c][i] & 0x0F;
                let hi = qs_hi[c][i] & 0x0F;
                out[16 + c * 32 + i] = lo | (hi << 4);
            }
        }
        out
    }

    #[test]
    fn dot_q4_k_q8_0_matches_f32_reference_uniform_block() {
        // Build a super-block where dequant(w[i]) = 4.0 for every element:
        //   d = 1.0, dmin = 0.0, all sub_scales = 1, all sub_mins = 0,
        //   all q4 = 4 (low AND high nibbles).
        let q4k = build_q4k_superblock(
            1.0,
            0.0,
            [1u8; 8],
            [0u8; 8],
            &[[4u8; 32]; 4],
            &[[4u8; 32]; 4],
        );
        let deq = dequant_q4_k(&q4k, 256);
        for (i, v) in deq.iter().enumerate() {
            assert!((v - 4.0).abs() < 1e-5, "elem {i}: {v}");
        }

        // Input: all 2.0.  f32 dot = 256 × 2.0 × 4.0 = 2048.
        let input_f32: Vec<f32> = vec![2.0; 256];
        let input_q8 = quantize_f32_to_q8_0(&input_f32);
        let expected: f32 = deq.iter().zip(&input_f32).map(|(w, x)| w * x).sum();
        let got = dot_q4_k_q8_0(&q4k, &input_q8, 256);
        let rel_err = ((got - expected) / expected).abs();
        assert!(rel_err < 0.01, "got={got} expected={expected}");
    }

    #[test]
    fn dot_q4_k_q8_0_matches_f32_reference_varied_block() {
        // Varied per-sub-block scales/mins and non-uniform qs.
        let sub_scales = [3u8, 5, 7, 2, 10, 4, 6, 8];
        let sub_mins = [1u8, 2, 3, 1, 2, 1, 1, 2];
        // qs_lo[c][i] spans 0..=15 in a pattern; qs_hi similarly.
        let qs_lo: [[u8; 32]; 4] = std::array::from_fn(|c| {
            std::array::from_fn(|i| ((c * 4 + i / 8) % 16) as u8)
        });
        let qs_hi: [[u8; 32]; 4] = std::array::from_fn(|c| {
            std::array::from_fn(|i| ((c * 2 + i % 16) % 16) as u8)
        });
        let q4k = build_q4k_superblock(0.03, 0.01, sub_scales, sub_mins, &qs_lo, &qs_hi);
        let deq = dequant_q4_k(&q4k, 256);

        // Mixed-sign input: small-amplitude cosine-like values.
        let input_f32: Vec<f32> = (0..256)
            .map(|i| ((i as f32 * 0.1).sin() * 0.5))
            .collect();
        let input_q8 = quantize_f32_to_q8_0(&input_f32);

        let expected: f32 = deq.iter().zip(&input_f32).map(|(w, x)| w * x).sum();
        let got = dot_q4_k_q8_0(&q4k, &input_q8, 256);

        // Q8_0 quantization of the *input* has ~0.4% max relative error per
        // block; combined with the Q4_K weights this propagates to ~1% worst
        // case.  Allow 2% relative tolerance.
        let abs_err = (got - expected).abs();
        let ref_mag = expected.abs().max(1e-6);
        assert!(
            abs_err / ref_mag < 0.02,
            "got={got} expected={expected} rel_err={}",
            abs_err / ref_mag
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dot_q4_k_q8_0_neon_matches_scalar_uniform() {
        // Same uniform-block setup as the scalar test — kernel must produce
        // bit-identical (or within fp rounding) result.
        let q4k = build_q4k_superblock(
            1.0,
            0.0,
            [1u8; 8],
            [0u8; 8],
            &[[4u8; 32]; 4],
            &[[4u8; 32]; 4],
        );
        let input_f32: Vec<f32> = vec![2.0; 256];
        let input_q8 = quantize_f32_to_q8_0(&input_f32);
        let scalar = dot_q4_k_q8_0(&q4k, &input_q8, 256);
        let neon = dot_q4_k_q8_0_neon(&q4k, &input_q8, 256);
        assert!(
            (scalar - neon).abs() < 1e-3,
            "NEON {neon} ≠ scalar {scalar}"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dot_q4_k_q8_0_neon_matches_scalar_varied() {
        // Same varied-block setup as the scalar test.
        let sub_scales = [3u8, 5, 7, 2, 10, 4, 6, 8];
        let sub_mins = [1u8, 2, 3, 1, 2, 1, 1, 2];
        let qs_lo: [[u8; 32]; 4] = std::array::from_fn(|c| {
            std::array::from_fn(|i| ((c * 4 + i / 8) % 16) as u8)
        });
        let qs_hi: [[u8; 32]; 4] = std::array::from_fn(|c| {
            std::array::from_fn(|i| ((c * 2 + i % 16) % 16) as u8)
        });
        let q4k = build_q4k_superblock(0.03, 0.01, sub_scales, sub_mins, &qs_lo, &qs_hi);
        let input_f32: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let input_q8 = quantize_f32_to_q8_0(&input_f32);

        let scalar = dot_q4_k_q8_0(&q4k, &input_q8, 256);
        let neon = dot_q4_k_q8_0_neon(&q4k, &input_q8, 256);
        // Both go through the same Q8_0 input + same Q4_K weights; results
        // should agree to fp32 rounding.
        let abs = (scalar - neon).abs();
        let rel = abs / scalar.abs().max(1e-6);
        assert!(rel < 1e-4, "NEON={neon} scalar={scalar} rel_err={rel}");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dot_q4_k_q8_0_neon_multi_superblock() {
        // K=512: two super-blocks back-to-back.  Catches off-by-superblock
        // pointer math errors.
        let sb1 = build_q4k_superblock(
            0.05, 0.02,
            [4, 8, 12, 16, 20, 24, 28, 32],
            [1, 2, 3, 4, 5, 6, 7, 8],
            &[[2u8; 32], [4u8; 32], [6u8; 32], [8u8; 32]],
            &[[3u8; 32], [5u8; 32], [7u8; 32], [9u8; 32]],
        );
        let sb2 = build_q4k_superblock(
            0.07, 0.03,
            [10, 11, 12, 13, 14, 15, 16, 17],
            [2, 3, 4, 5, 6, 7, 8, 9],
            &[[1u8; 32], [3u8; 32], [5u8; 32], [7u8; 32]],
            &[[2u8; 32], [4u8; 32], [6u8; 32], [8u8; 32]],
        );
        let mut q4k = sb1;
        q4k.extend_from_slice(&sb2);

        let input_f32: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).cos() * 0.3).collect();
        let input_q8 = quantize_f32_to_q8_0(&input_f32);

        let scalar = dot_q4_k_q8_0(&q4k, &input_q8, 512);
        let neon = dot_q4_k_q8_0_neon(&q4k, &input_q8, 512);
        let rel = ((scalar - neon).abs()) / scalar.abs().max(1e-6);
        assert!(rel < 1e-4, "scalar={scalar} neon={neon} rel_err={rel}");
    }

    #[test]
    fn quantize_f32_to_q4_k_round_trip_smooth() {
        // 512-element smooth ramp: two super-blocks' worth.
        let values: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.005).collect();
        let q4k = quantize_f32_to_q4_k(&values);
        assert_eq!(q4k.len(), 2 * 144);
        let recovered = dequant_q4_k(&q4k, values.len());

        // Per-sub-block max error ≤ scale/2 ≈ (max-min)/30 per sub-block.
        // Max amplitude span over 32 elements is ≈ 0.155, so max error ≈ 0.006.
        // Add some slack for the super-block packing quantization of scales
        // (6-bit scales lose up to 1/63 ≈ 1.6% relative).
        let max_err = values
            .iter()
            .zip(&recovered)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.05,
            "Q4_K round-trip max error too large: {max_err}"
        );
    }

    #[test]
    fn quantize_f32_to_q4_k_round_trip_zero_block() {
        // All-zero input should dequant back to zero (or at least near-zero).
        let values = vec![0.0f32; 256];
        let q4k = quantize_f32_to_q4_k(&values);
        let recovered = dequant_q4_k(&q4k, 256);
        for v in recovered {
            assert!(v.abs() < 1e-3, "zero block dequanted to {v}");
        }
    }

    #[test]
    fn quantize_f32_to_q8_0_round_trip() {
        // Smooth ramp across two blocks, including negatives and zeros.
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let q8_bytes = quantize_f32_to_q8_0(&values);
        assert_eq!(q8_bytes.len(), 2 * 34);

        let recovered = dequant_q8_0(&q8_bytes, values.len());
        assert_eq!(recovered.len(), values.len());

        // Per-block scale = absmax/127 → max abs error ≤ scale/2.
        // Worst-case absmax in this ramp is ~1.55, so error bound ~0.013.
        for (orig, got) in values.iter().zip(recovered.iter()) {
            let err = (orig - got).abs();
            assert!(err < 0.02, "round-trip error too large: {orig} → {got}");
        }
    }

    #[test]
    fn quantize_f32_to_q8_0_zero_block() {
        let values = vec![0.0f32; 32];
        let q8_bytes = quantize_f32_to_q8_0(&values);
        let recovered = dequant_q8_0(&q8_bytes, values.len());
        for v in recovered {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn bf16_conversion() {
        // BF16 1.0 = 0x3F80 (upper 16 bits of f32 1.0)
        let data = 0x3F80u16.to_le_bytes();
        let result = dequant_bf16(&data, 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dequant_f32_identity() {
        let values = vec![1.0f32, 2.0, -3.5, 0.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = dequant_f32(&bytes, 4);
        assert_eq!(result, values);
    }

    #[test]
    fn dequant_f16_roundtrip() {
        // Pack known f16 values and verify dequantization
        let f16_one = 0x3C00u16; // 1.0
        let f16_half = 0x3800u16; // 0.5
        let bytes: Vec<u8> = [f16_one, f16_half]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result = dequant_f16(&bytes, 2);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn split_fused_qkv_and_ffn_f32() {
        // One layer, MHA (heads==kv_heads), hidden=4, heads=2, head_dim=2, intermediate=6.
        // QKV total = hidden * (heads*head_dim + 2*kv_heads*head_dim) = 4*12 = 48.
        //   Q = [1..=16], K = [17..=32], V = [33..=48]
        // ffn total = hidden * 2 * intermediate = 4 * 2 * 6 = 48.
        //   gate = [1..=24], up = [25..=48]
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.qkv_proj.weight".to_string(),
            (1..=48).map(|x| x as f32).collect::<Vec<f32>>(),
        );
        let gate = (1..=24).map(|x| x as f32).collect::<Vec<f32>>();
        let up = (25..=48).map(|x| x as f32).collect::<Vec<f32>>();
        let fused_ffn = gate.iter().chain(up.iter()).copied().collect::<Vec<f32>>();
        tensors.insert("model.layers.0.mlp.up_proj.weight".to_string(), fused_ffn);
        let mut weights = ModelWeights { tensors };

        split_fused_tensors_f32(&mut weights, 1, 4, 2, 2, 2, 6);

        assert_eq!(
            weights
                .get("model.layers.0.self_attn.q_proj.weight")
                .unwrap()
                .len(),
            16
        );
        assert_eq!(
            weights
                .get("model.layers.0.self_attn.k_proj.weight")
                .unwrap()
                .len(),
            16
        );
        assert_eq!(
            weights
                .get("model.layers.0.self_attn.v_proj.weight")
                .unwrap()
                .len(),
            16
        );
        assert_eq!(
            weights
                .get("model.layers.0.self_attn.q_proj.weight")
                .unwrap(),
            &(1..=16).map(|x| x as f32).collect::<Vec<f32>>()[..]
        );
        assert_eq!(
            weights
                .get("model.layers.0.self_attn.v_proj.weight")
                .unwrap(),
            &(33..=48).map(|x| x as f32).collect::<Vec<f32>>()[..]
        );
        assert!(weights
            .get("model.layers.0.self_attn.qkv_proj.weight")
            .is_none());

        assert_eq!(
            weights.get("model.layers.0.mlp.gate_proj.weight").unwrap(),
            &gate[..]
        );
        assert_eq!(
            weights.get("model.layers.0.mlp.up_proj.weight").unwrap(),
            &up[..]
        );
    }

    #[test]
    fn split_fused_noop_on_llama_layout() {
        // Llama has separate q/k/v and gate/up already; split should be a no-op.
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![1.0f32; 8],
        );
        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            vec![2.0f32; 24],
        );
        tensors.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            vec![3.0f32; 24],
        );
        let mut weights = ModelWeights {
            tensors: tensors.clone(),
        };
        split_fused_tensors_f32(&mut weights, 1, 4, 2, 2, 2, 6);
        // Nothing added or removed.
        assert_eq!(weights.tensors.len(), tensors.len());
        assert_eq!(
            weights
                .get("model.layers.0.mlp.up_proj.weight")
                .unwrap()
                .len(),
            24
        );
    }

    #[test]
    fn split_fused_qkv_q8_block_boundaries() {
        // 32 elements per Q8_0 block (34 bytes). Build a fused QKV tensor where
        // Q, K, V are each one block — ensures our byte-level split lands on
        // block boundaries.
        let block_bytes: [u8; 34] = {
            let mut b = [0u8; 34];
            b[0] = 0x00;
            b[1] = 0x3C; // f16 scale = 1.0
            for i in 0..32 {
                b[2 + i] = i as u8;
            }
            b
        };
        let mut q_block = block_bytes;
        let mut k_block = block_bytes;
        let mut v_block = block_bytes;
        q_block[2] = 1;
        k_block[2] = 2;
        v_block[2] = 3;
        let mut fused = Vec::with_capacity(34 * 3);
        fused.extend_from_slice(&q_block);
        fused.extend_from_slice(&k_block);
        fused.extend_from_slice(&v_block);

        let mut tensors = HashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.qkv_proj.weight".to_string(),
            WeightData::Q8_0Raw(fused),
        );
        let mut weights = ModelWeightsRaw { tensors };
        // hidden=1, heads=kv_heads=1, head_dim=32, intermediate=0 → Q=K=V= 32 elems each.
        split_fused_tensors(&mut weights, 1, 1, 1, 1, 32, 0);

        let q = weights
            .get_q8_raw("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        let k = weights
            .get_q8_raw("model.layers.0.self_attn.k_proj.weight")
            .unwrap();
        let v = weights
            .get_q8_raw("model.layers.0.self_attn.v_proj.weight")
            .unwrap();
        assert_eq!(q.len(), 34);
        assert_eq!(k.len(), 34);
        assert_eq!(v.len(), 34);
        assert_eq!(q[2], 1);
        assert_eq!(k[2], 2);
        assert_eq!(v[2], 3);
    }

    #[test]
    fn dequant_q8_0_basic() {
        // Build a Q8_0 block: scale=1.0 (f16), 32 int8 values
        let scale_f16: u16 = 0x3C00; // 1.0 in f16
        let mut block = Vec::new();
        block.extend_from_slice(&scale_f16.to_le_bytes());
        for i in 0..32 {
            block.push(i as u8); // int8 values 0..31
        }

        let result = dequant_q8_0(&block, 32);
        assert_eq!(result.len(), 32);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[31] - 31.0).abs() < 1e-6);
    }

    #[test]
    fn dequant_q4_0_basic() {
        // Build a Q4_0 block: scale=1.0 (f16), 16 bytes of packed 4-bit values
        let scale_f16: u16 = 0x3C00; // 1.0
        let mut block = Vec::new();
        block.extend_from_slice(&scale_f16.to_le_bytes());
        // Pack 32 values: lo nibble = low 16 elements, hi nibble = high 16 elements
        // Use value 8 (which becomes 0 after subtracting offset of 8)
        block.extend(std::iter::repeat_n(0x88u8, 16));

        let result = dequant_q4_0(&block, 32);
        assert_eq!(result.len(), 32);
        // All values should be 0 (8 - 8 = 0, * 1.0 = 0.0)
        for val in &result {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn dequant_q4_1_basic() {
        // Build a Q4_1 block: scale=2.0 (f16), min=1.0 (f16), 16 bytes packed
        let scale_f16: u16 = 0x4000; // 2.0
        let min_f16: u16 = 0x3C00; // 1.0
        let mut block = Vec::new();
        block.extend_from_slice(&scale_f16.to_le_bytes());
        block.extend_from_slice(&min_f16.to_le_bytes());
        // All zeros → value = 0 * 2.0 + 1.0 = 1.0
        block.extend(std::iter::repeat_n(0x00u8, 16));

        let result = dequant_q4_1(&block, 32);
        assert_eq!(result.len(), 32);
        for val in &result {
            assert!((val - 1.0).abs() < 1e-6, "expected 1.0, got {val}");
        }
    }

    #[test]
    fn model_weights_accessors() {
        let mut tensors = HashMap::new();
        tensors.insert("w1".to_string(), vec![1.0f32; 100]);
        tensors.insert("w2".to_string(), vec![2.0f32; 200]);
        let weights = ModelWeights { tensors };

        assert_eq!(weights.len(), 2);
        assert!(!weights.is_empty());
        assert_eq!(weights.total_elements(), 300);
        assert_eq!(weights.memory_bytes(), 1200);
        assert_eq!(weights.get("w1").unwrap().len(), 100);
        assert_eq!(weights.tensor("w2").len(), 200);
    }

    #[test]
    fn model_weights_raw_accessors() {
        let mut tensors = HashMap::new();
        tensors.insert("norm.weight".to_string(), WeightData::F32(vec![1.0f32; 64]));
        tensors.insert(
            "q_proj.weight".to_string(),
            WeightData::Q8_0Raw(vec![0u8; 68]),
        ); // 2 blocks * 34 bytes
        let raw = ModelWeightsRaw { tensors };

        assert_eq!(raw.len(), 2);
        assert!(!raw.is_empty());
        assert!(raw.get_f32("norm.weight").is_some());
        assert_eq!(raw.get_f32("norm.weight").unwrap().len(), 64);
        assert!(raw.get_q8_raw("q_proj.weight").is_some());
        assert_eq!(raw.get_q8_raw("q_proj.weight").unwrap().len(), 68);
        // Cross-type accessors should return None
        assert!(raw.get_f32("q_proj.weight").is_none());
        assert!(raw.get_q8_raw("norm.weight").is_none());
        // Memory bytes: 64*4 + 68 = 256 + 68 = 324
        assert_eq!(raw.memory_bytes(), 64 * 4 + 68);
    }

    #[test]
    fn model_weights_raw_q4_0_accessor() {
        let mut tensors = HashMap::new();
        tensors.insert("norm.weight".to_string(), WeightData::F32(vec![1.0f32; 64]));
        // Q4_0: 2 blocks * 18 bytes each = 36 bytes (64 elements / 32 per block * 18)
        tensors.insert(
            "q_proj.weight".to_string(),
            WeightData::Q4_0Raw(vec![0u8; 36]),
        );
        let raw = ModelWeightsRaw { tensors };

        assert_eq!(raw.len(), 2);
        assert!(raw.get_q4_raw("q_proj.weight").is_some());
        assert_eq!(raw.get_q4_raw("q_proj.weight").unwrap().len(), 36);
        // Cross-type accessors should return None
        assert!(raw.get_f32("q_proj.weight").is_none());
        assert!(raw.get_q8_raw("q_proj.weight").is_none());
        assert!(raw.get_q4_raw("norm.weight").is_none());
        // Memory bytes: 64*4 + 36 = 256 + 36 = 292
        assert_eq!(raw.memory_bytes(), 64 * 4 + 36);
    }

    #[test]
    fn weight_data_q4_0_raw_stored_correctly() {
        // Build a minimal Q4_0 block (18 bytes)
        let scale_f16: u16 = 0x3C00; // 1.0 in f16
        let mut block = Vec::new();
        block.extend_from_slice(&scale_f16.to_le_bytes()); // 2 bytes scale
        block.extend(std::iter::repeat_n(0x88u8, 16)); // 16 bytes: all nibbles = 8 → 8-8=0

        let wd = WeightData::Q4_0Raw(block.clone());
        match &wd {
            WeightData::Q4_0Raw(v) => {
                assert_eq!(v.len(), 18);
                assert_eq!(&v[..], &block[..]);
            }
            _ => panic!("expected Q4_0Raw variant"),
        }
    }

    #[test]
    fn quantize_q4_0_roundtrip() {
        // Generate 64 f32 values (2 Q4_0 blocks)
        let input: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let q4_bytes = quantize_f32_to_q4_0(&input);
        assert_eq!(q4_bytes.len(), 2 * 18); // 2 blocks × 18 bytes

        // Dequantize and check approximate roundtrip
        let output = dequant_q4_0(&q4_bytes, 64);
        assert_eq!(output.len(), 64);

        // Q4_0 has only 4-bit precision (16 levels), so quantization error
        // can be up to ~1 step = scale/1. Check that the max error is bounded.
        let max_abs = input.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
        let step = max_abs / 8.0; // Q4_0 scale
        for (i, (&orig, &decoded)) in input.iter().zip(output.iter()).enumerate() {
            let diff = (orig - decoded).abs();
            assert!(
                diff <= step + 0.01,
                "element {i}: orig={orig}, decoded={decoded}, diff={diff}, step={step}"
            );
        }
    }

    #[test]
    fn quantize_q4_0_zeros() {
        let input = vec![0.0f32; 32];
        let q4_bytes = quantize_f32_to_q4_0(&input);
        assert_eq!(q4_bytes.len(), 18);

        // Scale should be 0
        let scale_bits = u16::from_le_bytes([q4_bytes[0], q4_bytes[1]]);
        assert_eq!(scale_bits, 0, "zero input should produce zero scale");

        // Dequantized values should all be 0
        let output = dequant_q4_0(&q4_bytes, 32);
        for &v in &output {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn get_scale_min_k4_low_groups() {
        // For groups 0..3, scale = scales[j] & 63, min = scales[j+4] & 63.
        let mut scales = [0u8; 12];
        scales[0] = 10; // scale for group 0
        scales[4] = 20; // min for group 0
        scales[1] = 30; // scale for group 1
        scales[5] = 40; // min for group 1
        scales[3] = 63; // scale for group 3 (max 6-bit value)
        scales[7] = 63; // min for group 3

        let (sc, m) = get_scale_min_k4(0, &scales);
        assert_eq!(sc, 10);
        assert_eq!(m, 20);

        let (sc, m) = get_scale_min_k4(1, &scales);
        assert_eq!(sc, 30);
        assert_eq!(m, 40);

        let (sc, m) = get_scale_min_k4(3, &scales);
        assert_eq!(sc, 63);
        assert_eq!(m, 63);
    }

    #[test]
    fn get_scale_min_k4_high_groups() {
        // For groups 4..7: scale = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
        //                   min   = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
        let mut scales = [0u8; 12];
        // Group 4: j=4, scale from scales[8] low nibble + scales[0] top 2 bits
        //          min from scales[8] high nibble + scales[4] top 2 bits
        scales[8] = 0xA5; // low nibble=5 (for scale), high nibble=0xA=10 (for min)
        scales[0] = 0xC0; // top 2 bits = 3 -> scale high bits = 3 << 4 = 48
        scales[4] = 0x80; // top 2 bits = 2 -> min high bits = 2 << 4 = 32

        let (sc, m) = get_scale_min_k4(4, &scales);
        assert_eq!(sc, 5 | (3 << 4)); // 5 + 48 = 53
        assert_eq!(m, 10 | (2 << 4)); // 10 + 32 = 42
    }

    #[test]
    fn dequant_q4_k_basic() {
        // Build a Q4_K block (144 bytes for 256 elements).
        // d=1.0, dmin=0.0, all scales=1, all mins=0, all qs=0.
        // Expected: d * 1 * 0 - 0 * 0 = 0.0 for all elements.
        let mut block = vec![0u8; 144];
        let d_f16: u16 = 0x3C00; // 1.0
        let dmin_f16: u16 = 0x0000; // 0.0
        block[0..2].copy_from_slice(&d_f16.to_le_bytes());
        block[2..4].copy_from_slice(&dmin_f16.to_le_bytes());
        // Set all 8 scales to 1 (6-bit value 1 in the packed format).
        // Groups 0..3: scales[j] = 1
        block[4] = 1;
        block[5] = 1;
        block[6] = 1;
        block[7] = 1;
        // Groups 0..3 mins: scales[j+4] = 0 (already zero)
        // Groups 4..7: scales[j+4] low nibble = 1
        block[12] = 0x01;
        block[13] = 0x01;
        block[14] = 0x01;
        block[15] = 0x01;
        // qs at offset 16..144, all zeros (q=0).

        let result = dequant_q4_k(&block, 256);
        assert_eq!(result.len(), 256);
        for (i, &val) in result.iter().enumerate() {
            assert!(val.abs() < 1e-6, "element {i}: expected 0.0, got {val}");
        }
    }

    #[test]
    fn dequant_q4_k_with_values() {
        // Build a Q4_K block where the first 32 elements use low nibble.
        // d=2.0, dmin=1.0, group 0 scale=3, group 0 min=2.
        // qs[0..31] low nibble = 5 → value = 2.0 * 3 * 5 - 1.0 * 2 = 30 - 2 = 28.
        let mut block = vec![0u8; 144];
        let d_f16: u16 = 0x4000; // 2.0
        let dmin_f16: u16 = 0x3C00; // 1.0
        block[0..2].copy_from_slice(&d_f16.to_le_bytes());
        block[2..4].copy_from_slice(&dmin_f16.to_le_bytes());
        // Group 0: scales[0] = 3 (6-bit), mins (scales[4]) = 2
        block[4] = 3;
        block[8] = 2;
        // qs at offset 16. First 32 elements use low nibble of qs[0..31].
        for i in 0..32 {
            block[16 + i] = 0x05; // low nibble = 5, high nibble = 0
        }

        let result = dequant_q4_k(&block, 256);
        // First 32 elements: d * sc * q - dmin * m = 2.0 * 3 * 5 - 1.0 * 2 = 28.0
        for (i, &v) in result.iter().take(32).enumerate() {
            assert!(
                (v - 28.0).abs() < 1e-4,
                "element {i}: expected 28.0, got {v}"
            );
        }
    }

    #[test]
    fn dequant_q5_k_basic_zeros() {
        // Build a Q5_K block (176 bytes) with all zeros.
        // d=1.0, dmin=0.0, scales all 1, qh=0, qs=0.
        // Expected: d * 1 * 0 - 0 * 0 = 0.0
        let mut block = vec![0u8; 176];
        let d_f16: u16 = 0x3C00; // 1.0
        let dmin_f16: u16 = 0x0000; // 0.0
        block[0..2].copy_from_slice(&d_f16.to_le_bytes());
        block[2..4].copy_from_slice(&dmin_f16.to_le_bytes());
        // Set scales for groups 0..3 to 1
        block[4] = 1;
        block[5] = 1;
        block[6] = 1;
        block[7] = 1;
        // Groups 4..7: scales[j+4] low nibble = 1
        block[12] = 0x01;
        block[13] = 0x01;
        block[14] = 0x01;
        block[15] = 0x01;

        let result = dequant_q5_k(&block, 256);
        assert_eq!(result.len(), 256);
        for (i, &val) in result.iter().enumerate() {
            assert!(val.abs() < 1e-6, "element {i}: expected 0.0, got {val}");
        }
    }

    #[test]
    fn dequant_q5_k_with_high_bit() {
        // Test that the 5th bit from qh is correctly incorporated.
        // Build a Q5_K block. d=1.0, dmin=0.0, group 0 scale=1.
        // Set qs[0] low nibble = 3 and qh bit for element 0 = 1.
        // q5 value = 3 + 16 = 19. Output = 1.0 * 1 * 19 = 19.0.
        let mut block = vec![0u8; 176];
        let d_f16: u16 = 0x3C00; // 1.0
        block[0..2].copy_from_slice(&d_f16.to_le_bytes());
        // dmin = 0
        block[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
        // Group 0 scale = 1
        block[4] = 1;
        // qh at offset 16..48. Set bit 0 of qh[0] (u1=1 for first chunk, element 0).
        block[16] = 0x01; // qh[0] bit 0 set
                          // qs at offset 48..176. qs[0] low nibble = 3.
        block[48] = 0x03;

        let result = dequant_q5_k(&block, 256);
        // Element 0: d * scale * (3 + 16) = 1.0 * 1 * 19 = 19.0
        assert!(
            (result[0] - 19.0).abs() < 1e-4,
            "element 0: expected 19.0, got {}",
            result[0]
        );
    }

    #[test]
    fn dequant_q5_k_high_nibble_with_qh() {
        // Test the high nibble path (elements 32..63 in first chunk).
        // d=1.0, dmin=0.0, group 1 scale=2.
        // qs[0] high nibble = 7, qh[0] bit 1 = 1 (u2=2 for first chunk).
        // q5 value = 7 + 16 = 23. Output = 1.0 * 2 * 23 = 46.0.
        let mut block = vec![0u8; 176];
        let d_f16: u16 = 0x3C00; // 1.0
        block[0..2].copy_from_slice(&d_f16.to_le_bytes());
        block[2..4].copy_from_slice(&0x0000u16.to_le_bytes());
        // Group 1 scale = 2 (scales[1] for groups < 4)
        block[5] = 2;
        // qh at offset 16..48. Set bit 1 of qh[0] (u2=2 for first chunk, element 32).
        block[16] = 0x02;
        // qs at offset 48..176. qs[0] high nibble = 7.
        block[48] = 0x70;

        let result = dequant_q5_k(&block, 256);
        // Element 32: d * scale * (7 + 16) = 1.0 * 2 * 23 = 46.0
        assert!(
            (result[32] - 46.0).abs() < 1e-4,
            "element 32: expected 46.0, got {}",
            result[32]
        );
    }

    #[test]
    fn dequant_q6_k_basic_zeros() {
        // Build a Q6_K block (210 bytes) with all zeros.
        // d=1.0, all scales=1, all ql/qh=0.
        // q6 = 0 - 32 = -32. Output = 1.0 * 1 * (-32) = -32.0.
        let mut block = vec![0u8; 210];
        let d_f16: u16 = 0x3C00; // 1.0
        block[208..210].copy_from_slice(&d_f16.to_le_bytes());
        // scales at offset 192..208 (int8). Set all to 1.
        for b in block.iter_mut().take(208).skip(192) {
            *b = 1;
        }

        let result = dequant_q6_k(&block, 256);
        assert_eq!(result.len(), 256);
        // All q6 values are 0, so q = 0 - 32 = -32. Output = 1.0 * 1 * -32 = -32.
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - (-32.0)).abs() < 1e-4,
                "element {i}: expected -32.0, got {val}"
            );
        }
    }

    #[test]
    fn dequant_q6_k_with_values() {
        // Build a Q6_K block. d=0.5, scale[0]=2.
        // Element 0: ql[0] low nibble = 0xA = 10, qh[0] bits 0-1 = 2.
        // q6 = (10 | (2 << 4)) - 32 = (10 | 32) - 32 = 42 - 32 = 10.
        // Output = 0.5 * 2 * 10 = 10.0.
        let mut block = vec![0u8; 210];
        let d_f16: u16 = 0x3800; // 0.5
        block[208..210].copy_from_slice(&d_f16.to_le_bytes());
        // scales at offset 192..208. scale[0] = 2 (as int8).
        block[192] = 2;
        // ql at offset 0..128. ql[0] = 0xAA (low nibble = 0xA for q1, value doesn't matter for q3 high nibble).
        block[0] = 0x0A; // low nibble = 10, high nibble = 0
                         // qh at offset 128..192. qh[0] bits 0-1 = 2.
        block[128] = 0x02;

        let result = dequant_q6_k(&block, 256);
        // Element 0 (q1): (0xA | ((0x02 & 3) << 4)) - 32 = (10 | 32) - 32 = 10
        // Output = 0.5 * 2 * 10 = 10.0
        assert!(
            (result[0] - 10.0).abs() < 1e-4,
            "element 0: expected 10.0, got {}",
            result[0]
        );
    }

    #[test]
    fn dequant_q6_k_four_interleaved_values() {
        // Verify all four interleaved values (q1, q2, q3, q4) from one iteration.
        // d=1.0, all scales=1.
        // l=0, first 128-element chunk.
        let mut block = vec![0u8; 210];
        let d_f16: u16 = 0x3C00; // 1.0
        block[208..210].copy_from_slice(&d_f16.to_le_bytes());
        // All scales = 1 (int8)
        for b in block.iter_mut().take(208).skip(192) {
            *b = 1;
        }
        // ql[0] = 0x31 → low nibble = 1 (for q1), high nibble = 3 (for q3)
        block[0] = 0x31;
        // ql[32] = 0x52 → low nibble = 2 (for q2), high nibble = 5 (for q4)
        block[32] = 0x52;
        // qh[0] = 0b_11_10_01_00 = 0xE4
        //   bits 0-1 = 0 (for q1), bits 2-3 = 1 (for q2),
        //   bits 4-5 = 2 (for q3), bits 6-7 = 3 (for q4)
        block[128] = 0xE4;

        let result = dequant_q6_k(&block, 256);
        // q1 = (1 | (0 << 4)) - 32 = 1 - 32 = -31 → output = 1.0 * 1 * -31 = -31.0
        assert!(
            (result[0] - (-31.0)).abs() < 1e-4,
            "q1 at [0]: expected -31.0, got {}",
            result[0]
        );
        // q2 = (2 | (1 << 4)) - 32 = 18 - 32 = -14 → output = 1.0 * 1 * -14 = -14.0
        assert!(
            (result[32] - (-14.0)).abs() < 1e-4,
            "q2 at [32]: expected -14.0, got {}",
            result[32]
        );
        // q3 = (3 | (2 << 4)) - 32 = 35 - 32 = 3 → output = 1.0 * 1 * 3 = 3.0
        assert!(
            (result[64] - 3.0).abs() < 1e-4,
            "q3 at [64]: expected 3.0, got {}",
            result[64]
        );
        // q4 = (5 | (3 << 4)) - 32 = 53 - 32 = 21 → output = 1.0 * 1 * 21 = 21.0
        assert!(
            (result[96] - 21.0).abs() < 1e-4,
            "q4 at [96]: expected 21.0, got {}",
            result[96]
        );
    }

    // ── Real-world validation tests ──────────────────────────────────────

    #[test]
    fn quantize_q4_0_handles_large_values() {
        // Values near the extremes of typical weight ranges.
        // Real model weights rarely exceed ~10.0 but we test larger values
        // to ensure no overflow in the quantization path.
        let mut input = vec![0.0f32; 32];
        input[0] = 1000.0;
        input[1] = -1000.0;
        input[15] = 500.0;
        input[16] = -500.0;

        let q4_bytes = quantize_f32_to_q4_0(&input);
        assert_eq!(q4_bytes.len(), 18); // 1 block

        // Dequantize and verify no NaN/Inf
        let output = dequant_q4_0(&q4_bytes, 32);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "dequantized value at index {i} is not finite: {v}"
            );
        }

        // The largest positive value should still be positive after roundtrip
        assert!(
            output[0] > 0.0,
            "large positive value should remain positive after Q4_0 roundtrip"
        );
        // The largest negative value should still be negative
        assert!(
            output[1] < 0.0,
            "large negative value should remain negative after Q4_0 roundtrip"
        );
    }

    #[test]
    fn dequant_q8_0_roundtrip_preserves_sign() {
        // Build Q8_0 blocks with mixed positive and negative values.
        // After dequantization, the sign must be preserved.
        //
        // Q8_0: each block = 2 bytes (f16 scale) + 32 bytes (int8 quantized values)
        // int8 values: -128 to 127 (signed)
        let scale_f16: u16 = 0x4000; // 2.0 in f16
        let mut block = Vec::new();
        block.extend_from_slice(&scale_f16.to_le_bytes());

        // Push signed int8 values: alternating positive and negative
        for i in 0..32i8 {
            if i % 2 == 0 {
                block.push(i as u8); // positive: 0, 2, 4, ...
            } else {
                block.push((-i) as u8); // negative: -1, -3, -5, ...
            }
        }

        let result = dequant_q8_0(&block, 32);
        assert_eq!(result.len(), 32);

        // Even indices should be non-negative
        for i in (0..32).step_by(2) {
            assert!(
                result[i] >= 0.0,
                "Q8_0 dequant: index {i} should be non-negative, got {}",
                result[i]
            );
        }
        // Odd indices should be negative (except i=0 which maps to 0)
        for i in (1..32).step_by(2) {
            assert!(
                result[i] < 0.0,
                "Q8_0 dequant: index {i} should be negative, got {}",
                result[i]
            );
        }

        // Verify specific values: val = int8_val * scale
        // index 0: 0 * 2.0 = 0.0
        assert!((result[0] - 0.0).abs() < 1e-6);
        // index 1: -1 * 2.0 = -2.0
        assert!((result[1] - (-2.0)).abs() < 1e-6);
        // index 2: 2 * 2.0 = 4.0
        assert!((result[2] - 4.0).abs() < 1e-6);
        // index 31: -31 * 2.0 = -62.0
        assert!((result[31] - (-62.0)).abs() < 1e-6);
    }

    #[test]
    fn dequant_q8_0_all_zeros() {
        // Zero scale means all dequantized values should be zero.
        let mut block = vec![0u8; 34]; // scale = 0, 32 arbitrary values
        for b in block.iter_mut().take(34).skip(2) {
            *b = 127; // max int8 value
        }

        let result = dequant_q8_0(&block, 32);
        for (i, &v) in result.iter().enumerate() {
            assert_eq!(v, 0.0, "with zero scale, index {i} should be 0.0, got {v}");
        }
    }

    #[test]
    fn quantize_q4_0_roundtrip_sign_preservation() {
        // Generate values with a clear sign pattern and verify the sign
        // survives the Q4_0 quantize → dequantize roundtrip.
        let input: Vec<f32> = (0..32)
            .map(|i| {
                if i < 16 {
                    -(i as f32 + 1.0)
                } else {
                    i as f32 - 15.0
                }
            })
            .collect();

        let q4_bytes = quantize_f32_to_q4_0(&input);
        let output = dequant_q4_0(&q4_bytes, 32);

        // First 16 values should be negative (or zero for small values due to quantization)
        for i in 0..16 {
            assert!(
                output[i] <= 0.0,
                "input[{i}]={}, roundtrip output[{i}]={} should be <= 0.0",
                input[i],
                output[i]
            );
        }
        // Last 16 values should be non-negative
        for i in 16..32 {
            assert!(
                output[i] >= 0.0,
                "input[{i}]={}, roundtrip output[{i}]={} should be >= 0.0",
                input[i],
                output[i]
            );
        }
    }
}
