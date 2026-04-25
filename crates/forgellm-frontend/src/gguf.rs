//! GGUF file format parser.
//!
//! GGUF (GGML Universal File) is the standard format for quantized LLM weights.
//! Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//!
//! File layout:
//! - Header: magic, version, tensor count, metadata kv count
//! - Metadata key-value pairs
//! - Tensor descriptors (name, shape, dtype, offset)
//! - Padding to alignment boundary
//! - Tensor data (contiguous, aligned)

use std::collections::HashMap;
use std::io::{self, Read, Seek};

use crate::ir::DType;

/// GGUF magic number: "GGUF" as bytes [0x47, 0x47, 0x55, 0x46] read as u32 LE.
const GGUF_MAGIC: u32 = 0x46554747;

/// Default alignment for tensor data.
const DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF metadata value types.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            MetadataValue::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// GGUF tensor type IDs (from the spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl GGMLType {
    pub fn from_u32(v: u32) -> Result<Self, GGUFError> {
        match v {
            0 => Ok(GGMLType::F32),
            1 => Ok(GGMLType::F16),
            2 => Ok(GGMLType::Q4_0),
            3 => Ok(GGMLType::Q4_1),
            6 => Ok(GGMLType::Q5_0),
            7 => Ok(GGMLType::Q5_1),
            8 => Ok(GGMLType::Q8_0),
            9 => Ok(GGMLType::Q8_1),
            10 => Ok(GGMLType::Q2K),
            11 => Ok(GGMLType::Q3K),
            12 => Ok(GGMLType::Q4K),
            13 => Ok(GGMLType::Q5K),
            14 => Ok(GGMLType::Q6K),
            15 => Ok(GGMLType::Q8K),
            16 => Ok(GGMLType::IQ2XXS),
            17 => Ok(GGMLType::IQ2XS),
            18 => Ok(GGMLType::IQ3XXS),
            19 => Ok(GGMLType::IQ1S),
            20 => Ok(GGMLType::IQ4NL),
            21 => Ok(GGMLType::IQ3S),
            22 => Ok(GGMLType::IQ2S),
            23 => Ok(GGMLType::IQ4XS),
            24 => Ok(GGMLType::I8),
            25 => Ok(GGMLType::I16),
            26 => Ok(GGMLType::I32),
            27 => Ok(GGMLType::I64),
            28 => Ok(GGMLType::F64),
            29 => Ok(GGMLType::IQ1M),
            30 => Ok(GGMLType::BF16),
            _ => Err(GGUFError::UnsupportedGGMLType(v)),
        }
    }

    /// Convert GGML type to our IR DType.
    ///
    /// Reflects the *post-load* dtype the rest of ForgeLLM sees, not the
    /// on-disk encoding.  Q4_0/Q8_0 are kept as raw bytes by the weight
    /// loader; everything else quantized goes through dequant → requant to
    /// Q8_0 so the existing Q8_0 codegen path handles it.  Dense f-types
    /// (F32/F16/BF16) stay dense.
    pub fn to_dtype(self) -> DType {
        match self {
            GGMLType::F32 => DType::F32,
            GGMLType::F16 => DType::F16,
            GGMLType::BF16 => DType::BF16,
            GGMLType::Q4_0 => DType::Q4_0,
            // Q4_K natively maps to DType::Q4_K — the loader keeps it as raw
            // 144-byte super-blocks when the target dtype is Q4_K.  Codegen
            // emits the native Q4_K kernel family for it.
            GGMLType::Q4K => DType::Q4_K,
            // Everything else quantized → Q8_0 post-load (re-quantized by the
            // weight loader).  In a Q4_K-target build, Q5_K / Q6_K / Q5_0 /
            // etc. are re-quantized to Q4_K instead, so a "majority Q4_K"
            // GGUF compiles uniformly through the Q4_K path.
            GGMLType::Q8_0
            | GGMLType::Q8_1
            | GGMLType::Q8K
            | GGMLType::Q4_1
            | GGMLType::IQ4NL
            | GGMLType::IQ4XS
            | GGMLType::Q5_0
            | GGMLType::Q5_1
            | GGMLType::Q5K
            | GGMLType::Q6K
            | GGMLType::Q2K
            | GGMLType::Q3K
            | GGMLType::IQ2XXS
            | GGMLType::IQ2XS
            | GGMLType::IQ2S
            | GGMLType::IQ3XXS
            | GGMLType::IQ3S
            | GGMLType::IQ1S
            | GGMLType::IQ1M => DType::Q8_0,
            GGMLType::I8 | GGMLType::I16 | GGMLType::I32 => DType::I32,
            GGMLType::I64 | GGMLType::F64 => DType::I64,
        }
    }

    /// Block size for this type (number of elements per quantized block).
    pub fn block_size(self) -> usize {
        match self {
            GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 | GGMLType::F64 => 1,
            GGMLType::I8 | GGMLType::I16 | GGMLType::I32 | GGMLType::I64 => 1,
            GGMLType::Q4_0 | GGMLType::Q4_1 => 32,
            GGMLType::Q5_0 | GGMLType::Q5_1 => 32,
            GGMLType::Q8_0 | GGMLType::Q8_1 => 32,
            GGMLType::Q2K
            | GGMLType::Q3K
            | GGMLType::Q4K
            | GGMLType::Q5K
            | GGMLType::Q6K
            | GGMLType::Q8K => 256,
            GGMLType::IQ2XXS | GGMLType::IQ2XS | GGMLType::IQ2S => 256,
            GGMLType::IQ3XXS | GGMLType::IQ3S => 256,
            GGMLType::IQ1S | GGMLType::IQ1M => 256,
            GGMLType::IQ4NL | GGMLType::IQ4XS => 32,
        }
    }

    /// Bytes per block for this type.
    pub fn type_size(self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 | GGMLType::BF16 => 2,
            GGMLType::F64 | GGMLType::I64 => 8,
            GGMLType::I32 => 4,
            GGMLType::I16 => 2,
            GGMLType::I8 => 1,
            GGMLType::Q4_0 => 18, // 32 * 4 bits / 8 + 2 (scale)
            GGMLType::Q4_1 => 20, // 32 * 4 bits / 8 + 2 (scale) + 2 (min)
            GGMLType::Q5_0 => 22, // 32 * 5 bits / 8 + 2 (scale) (rounded)
            GGMLType::Q5_1 => 24, // 32 * 5 bits / 8 + 2 + 2
            GGMLType::Q8_0 => 34, // 32 * 8 bits / 8 + 2 (scale)
            GGMLType::Q8_1 => 40, // 32 * 8 bits / 8 + 4 (scale) + 4 (min)
            GGMLType::Q2K => 84,  // 256 elements
            GGMLType::Q3K => 110,
            GGMLType::Q4K => 144,
            GGMLType::Q5K => 176,
            GGMLType::Q6K => 210,
            GGMLType::Q8K => 292,
            GGMLType::IQ2XXS => 66,
            GGMLType::IQ2XS => 74,
            GGMLType::IQ2S => 82,
            GGMLType::IQ3XXS => 98,
            GGMLType::IQ3S => 110,
            GGMLType::IQ1S => 50,
            GGMLType::IQ1M => 56,
            GGMLType::IQ4NL => 18,
            GGMLType::IQ4XS => 36,
        }
    }
}

/// Descriptor for a tensor within a GGUF file.
#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: GGMLType,
    /// Offset from the start of the tensor data section.
    pub offset: u64,
}

impl GGUFTensorInfo {
    /// Total number of elements in this tensor.
    pub fn numel(&self) -> u64 {
        self.dimensions.iter().product()
    }

    /// Total size in bytes of this tensor's data.
    pub fn data_size(&self) -> u64 {
        let n = self.numel() as usize;
        let block_size = self.ggml_type.block_size();
        let type_size = self.ggml_type.type_size();
        let num_blocks = n.div_ceil(block_size);
        (num_blocks * type_size) as u64
    }
}

/// Parsed GGUF file (metadata + tensor descriptors, no raw data).
#[derive(Debug, Clone)]
pub struct GGUFFile {
    pub version: u32,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<GGUFTensorInfo>,
    /// Byte offset where tensor data begins in the file.
    pub tensor_data_offset: u64,
    /// Alignment for tensor data.
    pub alignment: u64,
}

impl GGUFFile {
    /// Get a metadata string value by key.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    /// Get a metadata u32 value by key.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    /// Get a metadata u64 value, trying u32 fallback.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| match v {
            MetadataValue::Uint64(x) => Some(*x),
            MetadataValue::Uint32(x) => Some(*x as u64),
            _ => None,
        })
    }

    /// Get the length of a metadata array by key.
    pub fn get_array_len(&self, key: &str) -> Option<usize> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_array())
            .map(|a| a.len())
    }

    /// Get a metadata f32 value by key.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    /// Detect the model architecture from metadata.
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    /// Find a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&GGUFTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// Errors that can occur when parsing a GGUF file.
#[derive(Debug, thiserror::Error)]
pub enum GGUFError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("invalid GGUF magic number: got 0x{0:08X}")]
    InvalidMagic(u32),

    #[error("unsupported GGUF version: {0} (supported: 2, 3)")]
    UnsupportedVersion(u32),

    #[error("unsupported GGML type ID: {0}")]
    UnsupportedGGMLType(u32),

    #[error("unsupported metadata value type: {0}")]
    UnsupportedValueType(u32),

    #[error("invalid UTF-8 in string")]
    InvalidUtf8,

    #[error("tensor dimension count {0} exceeds maximum (4)")]
    TooManyDimensions(u32),
}

/// Reader helper for GGUF binary data.
struct GGUFReader<R: Read + Seek> {
    reader: R,
}

impl<R: Read + Seek> GGUFReader<R> {
    fn new(reader: R) -> Self {
        Self { reader }
    }

    fn read_u8(&mut self) -> Result<u8, GGUFError> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_i8(&mut self) -> Result<i8, GGUFError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GGUFError> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16, GGUFError> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> Result<u32, GGUFError> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32, GGUFError> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64, GGUFError> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64, GGUFError> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32, GGUFError> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64, GGUFError> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_bool(&mut self) -> Result<bool, GGUFError> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String, GGUFError> {
        let len = self.read_u64()? as usize;
        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|_| GGUFError::InvalidUtf8)
    }

    fn read_metadata_value(&mut self, value_type: u32) -> Result<MetadataValue, GGUFError> {
        match value_type {
            0 => Ok(MetadataValue::Uint8(self.read_u8()?)),
            1 => Ok(MetadataValue::Int8(self.read_i8()?)),
            2 => Ok(MetadataValue::Uint16(self.read_u16()?)),
            3 => Ok(MetadataValue::Int16(self.read_i16()?)),
            4 => Ok(MetadataValue::Uint32(self.read_u32()?)),
            5 => Ok(MetadataValue::Int32(self.read_i32()?)),
            6 => Ok(MetadataValue::Float32(self.read_f32()?)),
            7 => Ok(MetadataValue::Bool(self.read_bool()?)),
            8 => Ok(MetadataValue::String(self.read_string()?)),
            9 => {
                // Array
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.read_metadata_value(elem_type)?);
                }
                Ok(MetadataValue::Array(values))
            }
            10 => Ok(MetadataValue::Uint64(self.read_u64()?)),
            11 => Ok(MetadataValue::Int64(self.read_i64()?)),
            12 => Ok(MetadataValue::Float64(self.read_f64()?)),
            _ => Err(GGUFError::UnsupportedValueType(value_type)),
        }
    }

    fn position(&mut self) -> Result<u64, GGUFError> {
        Ok(self.reader.stream_position()?)
    }
}

/// Parse a GGUF file from a reader.
///
/// This reads the header, metadata, and tensor descriptors. It does NOT
/// read the actual tensor data — that's deferred until compilation time.
pub fn parse<R: Read + Seek>(reader: R) -> Result<GGUFFile, GGUFError> {
    let mut r = GGUFReader::new(reader);

    // Read header
    let magic = r.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(GGUFError::InvalidMagic(magic));
    }

    let version = r.read_u32()?;
    if version != 2 && version != 3 {
        return Err(GGUFError::UnsupportedVersion(version));
    }

    let tensor_count = r.read_u64()?;
    let metadata_kv_count = r.read_u64()?;

    // Read metadata
    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = r.read_string()?;
        let value_type = r.read_u32()?;
        let value = r.read_metadata_value(value_type)?;
        metadata.insert(key, value);
    }

    // Read tensor descriptors
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.read_string()?;
        let n_dims = r.read_u32()?;
        if n_dims > 4 {
            return Err(GGUFError::TooManyDimensions(n_dims));
        }
        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dimensions.push(r.read_u64()?);
        }
        let ggml_type = GGMLType::from_u32(r.read_u32()?)?;
        let offset = r.read_u64()?;

        tensors.push(GGUFTensorInfo {
            name,
            dimensions,
            ggml_type,
            offset,
        });
    }

    // Determine alignment
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u32())
        .map(|v| v as u64)
        .unwrap_or(DEFAULT_ALIGNMENT);

    // Calculate tensor data offset (current position, aligned up)
    let pos = r.position()?;
    let tensor_data_offset = pos.div_ceil(alignment) * alignment;

    Ok(GGUFFile {
        version,
        metadata,
        tensors,
        tensor_data_offset,
        alignment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Helper to build a minimal GGUF file in memory.
    fn build_gguf_bytes(
        version: u32,
        metadata: &[(&str, u32, &[u8])], // (key, type_id, raw_value_bytes)
        tensors: &[(&str, &[u64], u32, u64)], // (name, dims, ggml_type, offset)
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version
        buf.extend_from_slice(&version.to_le_bytes());
        // Tensor count
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        // Metadata count
        buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        // Metadata KV pairs
        for (key, type_id, value_bytes) in metadata {
            // Key string: length (u64) + bytes
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            // Value type
            buf.extend_from_slice(&type_id.to_le_bytes());
            // Value bytes
            buf.extend_from_slice(value_bytes);
        }

        // Tensor descriptors
        for (name, dims, ggml_type, offset) in tensors {
            // Name string
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            // n_dims
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            // dimensions
            for d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            // type
            buf.extend_from_slice(&ggml_type.to_le_bytes());
            // offset
            buf.extend_from_slice(&offset.to_le_bytes());
        }

        buf
    }

    fn make_string_value(s: &str) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&(s.len() as u64).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
        v
    }

    fn make_u32_value(val: u32) -> Vec<u8> {
        val.to_le_bytes().to_vec()
    }

    fn make_f32_value(val: f32) -> Vec<u8> {
        val.to_le_bytes().to_vec()
    }

    #[test]
    fn parse_minimal_gguf_v3() {
        let arch_val = make_string_value("llama");
        let bytes = build_gguf_bytes(
            3,
            &[("general.architecture", 8, &arch_val)],
            &[("token_embd.weight", &[32000, 2048], 1, 0)], // F16
        );

        let file = parse(Cursor::new(bytes)).unwrap();
        assert_eq!(file.version, 3);
        assert_eq!(file.architecture(), Some("llama"));
        assert_eq!(file.tensors.len(), 1);
        assert_eq!(file.tensors[0].name, "token_embd.weight");
        assert_eq!(file.tensors[0].dimensions, vec![32000, 2048]);
        assert_eq!(file.tensors[0].ggml_type, GGMLType::F16);
    }

    #[test]
    fn parse_with_multiple_tensors() {
        let arch_val = make_string_value("llama");
        let hidden_val = make_u32_value(2048);
        let eps_val = make_f32_value(1e-5);

        let bytes = build_gguf_bytes(
            3,
            &[
                ("general.architecture", 8, &arch_val),
                ("llama.embedding_length", 4, &hidden_val),
                ("llama.attention.layer_norm_rms_epsilon", 6, &eps_val),
            ],
            &[
                ("token_embd.weight", &[32000, 2048], 1, 0),       // F16
                ("blk.0.attn_q.weight", &[2048, 2048], 8, 4096),   // Q8_0
                ("blk.0.ffn_gate.weight", &[5632, 2048], 2, 8192), // Q4_0
            ],
        );

        let file = parse(Cursor::new(bytes)).unwrap();
        assert_eq!(file.tensors.len(), 3);
        assert_eq!(file.get_u32("llama.embedding_length"), Some(2048));
        assert_eq!(
            file.get_f32("llama.attention.layer_norm_rms_epsilon"),
            Some(1e-5)
        );

        let q_weight = file.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(q_weight.ggml_type, GGMLType::Q8_0);
        assert_eq!(q_weight.numel(), 2048 * 2048);
    }

    #[test]
    fn parse_v2() {
        let bytes = build_gguf_bytes(2, &[], &[]);
        let file = parse(Cursor::new(bytes)).unwrap();
        assert_eq!(file.version, 2);
        assert!(file.tensors.is_empty());
    }

    #[test]
    fn reject_invalid_magic() {
        let mut bytes = vec![0u8; 32];
        bytes[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let result = parse(Cursor::new(bytes));
        assert!(matches!(result, Err(GGUFError::InvalidMagic(0xDEADBEEF))));
    }

    #[test]
    fn reject_unsupported_version() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&99u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata count
        let result = parse(Cursor::new(bytes));
        assert!(matches!(result, Err(GGUFError::UnsupportedVersion(99))));
    }

    #[test]
    fn ggml_type_conversions() {
        assert_eq!(GGMLType::F32.to_dtype(), DType::F32);
        assert_eq!(GGMLType::F16.to_dtype(), DType::F16);
        assert_eq!(GGMLType::BF16.to_dtype(), DType::BF16);
        assert_eq!(GGMLType::Q4_0.to_dtype(), DType::Q4_0);
        assert_eq!(GGMLType::Q8_0.to_dtype(), DType::Q8_0);
        // Q4_K maps natively to DType::Q4_K; under a Q4_K-target build the
        // loader keeps it as raw super-blocks for the native kernel.
        assert_eq!(GGMLType::Q4K.to_dtype(), DType::Q4_K);
        // Higher-precision K-quants go to Q8_0 by default (or are
        // requantized to Q4_K under a Q4_K target — that's a loader concern).
        assert_eq!(GGMLType::Q5K.to_dtype(), DType::Q8_0);
        assert_eq!(GGMLType::Q6K.to_dtype(), DType::Q8_0);
        assert_eq!(GGMLType::Q3K.to_dtype(), DType::Q8_0);
        assert_eq!(GGMLType::Q2K.to_dtype(), DType::Q8_0);
    }

    #[test]
    fn tensor_data_size() {
        // F16 tensor: 32000 * 2048 elements * 2 bytes = 131,072,000
        let t = GGUFTensorInfo {
            name: "test".into(),
            dimensions: vec![32000, 2048],
            ggml_type: GGMLType::F16,
            offset: 0,
        };
        assert_eq!(t.numel(), 32000 * 2048);
        assert_eq!(t.data_size(), 32000 * 2048 * 2);

        // Q4_0 tensor: 2048 * 2048 elements / 32 block_size * 18 bytes per block
        let t2 = GGUFTensorInfo {
            name: "test2".into(),
            dimensions: vec![2048, 2048],
            ggml_type: GGMLType::Q4_0,
            offset: 0,
        };
        assert_eq!(t2.data_size(), (2048 * 2048 / 32) * 18);
    }

    #[test]
    fn alignment_calculation() {
        let bytes = build_gguf_bytes(3, &[], &[]);
        let file = parse(Cursor::new(bytes)).unwrap();
        assert_eq!(file.alignment, DEFAULT_ALIGNMENT);
        // tensor_data_offset should be aligned to DEFAULT_ALIGNMENT
        assert_eq!(file.tensor_data_offset % DEFAULT_ALIGNMENT, 0);
    }

    #[test]
    fn metadata_value_accessors() {
        let s = MetadataValue::String("hello".into());
        assert_eq!(s.as_str(), Some("hello"));
        assert_eq!(s.as_u32(), None);

        let u = MetadataValue::Uint32(42);
        assert_eq!(u.as_u32(), Some(42));
        assert_eq!(u.as_str(), None);

        let f = MetadataValue::Float32(1.5);
        assert_eq!(f.as_f32(), Some(1.5));

        let b = MetadataValue::Bool(true);
        assert_eq!(b.as_bool(), Some(true));

        let a = MetadataValue::Array(vec![MetadataValue::Uint32(1), MetadataValue::Uint32(2)]);
        assert_eq!(a.as_array().unwrap().len(), 2);
    }
}
