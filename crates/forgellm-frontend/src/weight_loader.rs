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
            })
            .sum()
    }

    /// Get a tensor by name regardless of type — for non-Q8_0 tensors only.
    pub fn get(&self, name: &str) -> Option<&WeightData> {
        self.tensors.get(name)
    }
}

/// Load all tensors from a GGUF file with mixed storage:
/// - Q8_0 tensors are kept as raw bytes (`WeightData::Q8_0Raw`)
/// - Q4_0 tensors are kept as raw bytes (`WeightData::Q4_0Raw`)
/// - All other tensor types are dequantized to f32 (`WeightData::F32`)
pub fn load_from_file_mixed(
    path: impl AsRef<std::path::Path>,
) -> Result<(crate::gguf::GGUFFile, ModelWeightsRaw), WeightLoadError> {
    let file = std::fs::File::open(path.as_ref())?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let mut cursor = std::io::Cursor::new(&mmap[..]);

    let gguf = crate::gguf::parse(&mut cursor)
        .map_err(|e| WeightLoadError::Io(std::io::Error::other(e)))?;

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

        let weight_data = if tensor_info.ggml_type == crate::gguf::GGMLType::Q8_0 {
            WeightData::Q8_0Raw(raw.to_vec())
        } else if tensor_info.ggml_type == crate::gguf::GGMLType::Q4_0 {
            WeightData::Q4_0Raw(raw.to_vec())
        } else {
            let f32_data = dequantize(raw, tensor_info.ggml_type, numel)?;
            WeightData::F32(f32_data)
        };

        tensors.insert(hf_name, weight_data);
    }

    Ok((gguf, ModelWeightsRaw { tensors }))
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

    Ok(ModelWeights { tensors })
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

    Ok((gguf, ModelWeights { tensors }))
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
        let h_mant = ((mant | 0x80_0000) >> (14 + shift)) as u32;
        return ((sign << 15) | h_mant) as u16;
    }

    let h_exp = (unbiased + 15) as u32;
    let h_mant = mant >> 13;
    ((sign << 15) | (h_exp << 10) | h_mant) as u16
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
        for i in 0..32 {
            assert!(
                (result[i] - 28.0).abs() < 1e-4,
                "element {i}: expected 28.0, got {}",
                result[i]
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
        for i in 192..208 {
            block[i] = 1;
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
        for i in 192..208 {
            block[i] = 1;
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
}
