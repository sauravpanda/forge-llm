//! Weight loader — reads and dequantizes tensor data from GGUF files.
//!
//! Loads tensor data into f32 buffers, handling dequantization for
//! quantized formats (Q4_0, Q8_0, etc.) and fp16→f32 conversion.

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use crate::gguf::{self, GGMLType, GGUFFile, GGUFTensorInfo};

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

/// Dequantize raw bytes to f32 based on GGML type.
fn dequantize(data: &[u8], ggml_type: GGMLType, numel: usize) -> Result<Vec<f32>, WeightLoadError> {
    match ggml_type {
        GGMLType::F32 => Ok(dequant_f32(data, numel)),
        GGMLType::F16 => Ok(dequant_f16(data, numel)),
        GGMLType::BF16 => Ok(dequant_bf16(data, numel)),
        GGMLType::Q8_0 => Ok(dequant_q8_0(data, numel)),
        GGMLType::Q4_0 => Ok(dequant_q4_0(data, numel)),
        GGMLType::Q4_1 => Ok(dequant_q4_1(data, numel)),
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
}
