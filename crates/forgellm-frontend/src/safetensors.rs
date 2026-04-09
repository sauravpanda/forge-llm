//! SafeTensors file format parser.
//!
//! SafeTensors is a simple, safe format for storing tensors.
//! Format: 8-byte LE header size, then JSON header, then raw tensor data.
//!
//! The JSON header maps tensor names to {dtype, shape, data_offsets}.

use std::collections::HashMap;
use std::io::{self, Read, Seek};

use serde::Deserialize;

use crate::ir::DType;

/// Parsed SafeTensors file (metadata + tensor descriptors, no raw data).
#[derive(Debug, Clone)]
pub struct SafeTensorsFile {
    pub tensors: Vec<SafeTensorInfo>,
    /// Byte offset where tensor data begins (after the JSON header).
    pub data_offset: u64,
    /// Optional metadata from the header (e.g., format version).
    pub metadata: HashMap<String, String>,
}

/// Descriptor for a single tensor in a SafeTensors file.
#[derive(Debug, Clone)]
pub struct SafeTensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte range within the data section [start, end).
    pub data_start: usize,
    pub data_end: usize,
}

impl SafeTensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn data_size(&self) -> usize {
        self.data_end - self.data_start
    }
}

/// Raw JSON structure for a tensor entry in the SafeTensors header.
#[derive(Debug, Deserialize)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Errors that can occur when parsing a SafeTensors file.
#[derive(Debug, thiserror::Error)]
pub enum SafeTensorsError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("header size {0} exceeds maximum allowed (100MB)")]
    HeaderTooLarge(u64),

    #[error("unknown dtype: {0}")]
    UnknownDType(String),
}

/// Maximum header size (100MB) to prevent OOM on malformed files.
const MAX_HEADER_SIZE: u64 = 100 * 1024 * 1024;

/// Parse a SafeTensors dtype string to our IR DType.
fn parse_dtype(s: &str) -> Result<DType, SafeTensorsError> {
    match s {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "I32" => Ok(DType::I32),
        "I64" => Ok(DType::I64),
        "F64" => Ok(DType::I64), // No f64 in our IR; treat as i64 for storage size
        "U8" | "I8" | "BOOL" => Ok(DType::I32), // Small types mapped to i32
        "F8_E4M3" => Ok(DType::F8E4M3),
        "F8_E5M2" => Ok(DType::F8E5M2),
        _ => Err(SafeTensorsError::UnknownDType(s.to_string())),
    }
}

/// Parse a SafeTensors file from a reader.
///
/// Reads the header to get tensor metadata. Does NOT read tensor data.
pub fn parse<R: Read + Seek>(mut reader: R) -> Result<SafeTensorsFile, SafeTensorsError> {
    // Read 8-byte header size (little-endian u64)
    let mut size_buf = [0u8; 8];
    reader.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf);

    if header_size > MAX_HEADER_SIZE {
        return Err(SafeTensorsError::HeaderTooLarge(header_size));
    }

    // Read JSON header
    let mut header_buf = vec![0u8; header_size as usize];
    reader.read_exact(&mut header_buf)?;

    let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_buf)?;

    let mut tensors = Vec::new();
    let mut metadata = HashMap::new();

    for (key, value) in &raw {
        if key == "__metadata__" {
            // Extract metadata as string map
            if let Some(obj) = value.as_object() {
                for (mk, mv) in obj {
                    if let Some(s) = mv.as_str() {
                        metadata.insert(mk.clone(), s.to_string());
                    }
                }
            }
            continue;
        }

        let entry: RawTensorEntry = serde_json::from_value(value.clone())?;
        let dtype = parse_dtype(&entry.dtype)?;

        tensors.push(SafeTensorInfo {
            name: key.clone(),
            dtype,
            shape: entry.shape,
            data_start: entry.data_offsets[0],
            data_end: entry.data_offsets[1],
        });
    }

    // Sort tensors by data offset for deterministic ordering
    tensors.sort_by_key(|t| t.data_start);

    let data_offset = 8 + header_size;

    Ok(SafeTensorsFile {
        tensors,
        data_offset,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal SafeTensors file in memory.
    fn build_safetensors_bytes(header_json: &str) -> Vec<u8> {
        let header_bytes = header_json.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(header_bytes);
        // Append some dummy tensor data
        buf.extend_from_slice(&[0u8; 256]);
        buf
    }

    #[test]
    fn parse_simple_safetensors() {
        let header = r#"{
            "weight": {"dtype": "F16", "shape": [768, 768], "data_offsets": [0, 1179648]}
        }"#;
        let bytes = build_safetensors_bytes(header);

        let file = parse(Cursor::new(bytes)).unwrap();
        assert_eq!(file.tensors.len(), 1);
        assert_eq!(file.tensors[0].name, "weight");
        assert_eq!(file.tensors[0].dtype, DType::F16);
        assert_eq!(file.tensors[0].shape, vec![768, 768]);
        assert_eq!(file.tensors[0].data_start, 0);
        assert_eq!(file.tensors[0].data_end, 1179648);
    }

    #[test]
    fn parse_multiple_tensors() {
        let header = r#"{
            "__metadata__": {"format": "pt"},
            "model.embed_tokens.weight": {"dtype": "F32", "shape": [32000, 2048], "data_offsets": [0, 262144000]},
            "model.layers.0.self_attn.q_proj.weight": {"dtype": "F16", "shape": [2048, 2048], "data_offsets": [262144000, 270532608]}
        }"#;
        let bytes = build_safetensors_bytes(header);

        let file = parse(Cursor::new(bytes)).unwrap();
        assert_eq!(file.tensors.len(), 2);
        assert_eq!(file.metadata.get("format"), Some(&"pt".to_string()));

        // Should be sorted by data offset
        assert_eq!(file.tensors[0].name, "model.embed_tokens.weight");
        assert_eq!(file.tensors[0].dtype, DType::F32);
        assert_eq!(
            file.tensors[1].name,
            "model.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn parse_bf16_tensors() {
        let header =
            r#"{"w": {"dtype": "BF16", "shape": [1024, 512], "data_offsets": [0, 1048576]}}"#;
        let bytes = build_safetensors_bytes(header);

        let file = parse(Cursor::new(bytes)).unwrap();
        assert_eq!(file.tensors[0].dtype, DType::BF16);
    }

    #[test]
    fn reject_too_large_header() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(MAX_HEADER_SIZE + 1).to_le_bytes());
        let result = parse(Cursor::new(bytes));
        assert!(matches!(result, Err(SafeTensorsError::HeaderTooLarge(_))));
    }

    #[test]
    fn reject_unknown_dtype() {
        let header = r#"{"w": {"dtype": "COMPLEX128", "shape": [10], "data_offsets": [0, 100]}}"#;
        let bytes = build_safetensors_bytes(header);
        let result = parse(Cursor::new(bytes));
        assert!(matches!(result, Err(SafeTensorsError::UnknownDType(_))));
    }

    #[test]
    fn tensor_info_helpers() {
        let t = SafeTensorInfo {
            name: "test".into(),
            dtype: DType::F32,
            shape: vec![32, 64],
            data_start: 0,
            data_end: 8192,
        };
        assert_eq!(t.numel(), 2048);
        assert_eq!(t.data_size(), 8192);
    }

    #[test]
    fn data_offset_calculation() {
        let header = r#"{"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}"#;
        let header_len = header.len() as u64;
        let bytes = build_safetensors_bytes(header);

        let file = parse(Cursor::new(bytes)).unwrap();
        assert_eq!(file.data_offset, 8 + header_len);
    }
}
