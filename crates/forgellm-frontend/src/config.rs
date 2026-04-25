//! HuggingFace model config.json parser.
//!
//! Parses the `config.json` that accompanies SafeTensors model files
//! to extract architecture type and hyperparameters.

use serde::Deserialize;

use crate::ir::{Architecture, DType, ModelConfig};

/// Raw HuggingFace config.json structure.
/// Fields are optional since different architectures use different keys.
#[derive(Debug, Deserialize)]
pub struct HFConfig {
    /// Architecture identifiers
    pub model_type: Option<String>,
    pub architectures: Option<Vec<String>>,

    /// Core dimensions
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub vocab_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,

    /// Normalization
    pub rms_norm_eps: Option<f64>,
    pub layer_norm_eps: Option<f64>,
    pub layer_norm_epsilon: Option<f64>,

    /// RoPE
    pub rope_theta: Option<f64>,

    /// Data type
    pub torch_dtype: Option<String>,

    /// Sliding window attention size (Mistral).
    pub sliding_window: Option<usize>,
}

impl HFConfig {
    /// Parse from JSON bytes.
    pub fn from_json(json: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(json)
    }

    /// Detect the model architecture.
    pub fn detect_architecture(&self) -> Option<Architecture> {
        // Try model_type first
        if let Some(ref model_type) = self.model_type {
            match model_type.as_str() {
                "llama" => return Some(Architecture::Llama),
                "qwen2" => return Some(Architecture::Qwen2),
                "mistral" => return Some(Architecture::Mistral),
                "phi3" | "phi" => return Some(Architecture::Phi3),
                "gemma" | "gemma2" => return Some(Architecture::Gemma),
                "stablelm" | "stablelm_epoch" => return Some(Architecture::StableLM),
                _ => {}
            }
        }

        // Fall back to architectures list
        if let Some(ref archs) = self.architectures {
            for arch in archs {
                if arch.contains("Llama") {
                    return Some(Architecture::Llama);
                }
                if arch.contains("Qwen2") {
                    return Some(Architecture::Qwen2);
                }
                if arch.contains("Mistral") {
                    return Some(Architecture::Mistral);
                }
                if arch.contains("Phi3") || arch.contains("Phi") {
                    return Some(Architecture::Phi3);
                }
                if arch.contains("Gemma") {
                    return Some(Architecture::Gemma);
                }
                if arch.contains("StableLm") || arch.contains("StableLM") {
                    return Some(Architecture::StableLM);
                }
            }
        }

        None
    }

    /// Convert to a ModelConfig for the IR.
    pub fn to_model_config(&self) -> Option<ModelConfig> {
        let architecture = self.detect_architecture()?;
        let hidden_size = self.hidden_size?;
        let num_attention_heads = self.num_attention_heads?;

        let head_dim = self.head_dim.unwrap_or(hidden_size / num_attention_heads);

        let num_kv_heads = self.num_key_value_heads.unwrap_or(num_attention_heads);

        let norm_eps = self
            .rms_norm_eps
            .or(self.layer_norm_eps)
            .or(self.layer_norm_epsilon)
            .unwrap_or(1e-5);

        let dtype = self
            .torch_dtype
            .as_deref()
            .map(parse_torch_dtype)
            .unwrap_or(DType::F16);

        // Qwen2 uses bias on Q, K, V projections.
        let qkv_bias = matches!(architecture, Architecture::Qwen2);

        // Sliding window comes from config.json `sliding_window` field (Mistral).
        let sliding_window_size = self.sliding_window;

        // Gemma-1 uses approximate (tanh) GeLU in the FFN gated activation.
        let hidden_activation = match architecture {
            Architecture::Gemma => crate::ir::HiddenActivation::GeluApprox,
            _ => crate::ir::HiddenActivation::SiLU,
        };

        Some(ModelConfig {
            architecture,
            hidden_size,
            intermediate_size: self.intermediate_size.unwrap_or(hidden_size * 4),
            num_layers: self.num_hidden_layers?,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            vocab_size: self.vocab_size.unwrap_or(32000),
            max_seq_len: self.max_position_embeddings.unwrap_or(2048),
            rms_norm_eps: norm_eps as f32,
            rope_theta: self.rope_theta.unwrap_or(10000.0) as f32,
            dtype,
            lm_head_dtype: None,
            sliding_window_size,
            qkv_bias,
            hidden_activation,
        })
    }
}

fn parse_torch_dtype(s: &str) -> DType {
    match s {
        "float32" | "fp32" => DType::F32,
        "float16" | "fp16" => DType::F16,
        "bfloat16" | "bf16" => DType::BF16,
        _ => DType::F16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llama_config() {
        let json = r#"{
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "torch_dtype": "float16"
        }"#;

        let config = HFConfig::from_json(json.as_bytes()).unwrap();
        assert_eq!(config.detect_architecture(), Some(Architecture::Llama));

        let mc = config.to_model_config().unwrap();
        assert_eq!(mc.hidden_size, 2048);
        assert_eq!(mc.intermediate_size, 5632);
        assert_eq!(mc.num_layers, 16);
        assert_eq!(mc.num_attention_heads, 32);
        assert_eq!(mc.num_kv_heads, 8);
        assert_eq!(mc.head_dim, 64);
        assert_eq!(mc.vocab_size, 32000);
        assert_eq!(mc.dtype, DType::F16);
        assert!(!mc.qkv_bias);
        assert_eq!(mc.sliding_window_size, None);
    }

    #[test]
    fn parse_qwen2_config() {
        let json = r#"{
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "num_hidden_layers": 28,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            "max_position_embeddings": 32768,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "torch_dtype": "bfloat16"
        }"#;

        let config = HFConfig::from_json(json.as_bytes()).unwrap();
        assert_eq!(config.detect_architecture(), Some(Architecture::Qwen2));

        let mc = config.to_model_config().unwrap();
        assert_eq!(mc.hidden_size, 1536);
        assert_eq!(mc.num_kv_heads, 2);
        assert_eq!(mc.dtype, DType::BF16);
        assert_eq!(mc.head_dim, 128);
        // Qwen2 always sets qkv_bias = true
        assert!(mc.qkv_bias);
        assert_eq!(mc.sliding_window_size, None);
    }

    #[test]
    fn parse_mistral_config_with_sliding_window() {
        let json = r#"{
            "model_type": "mistral",
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "torch_dtype": "float16"
        }"#;

        let config = HFConfig::from_json(json.as_bytes()).unwrap();
        assert_eq!(config.detect_architecture(), Some(Architecture::Mistral));

        let mc = config.to_model_config().unwrap();
        assert_eq!(mc.architecture, Architecture::Mistral);
        assert_eq!(mc.sliding_window_size, Some(4096));
        assert!(!mc.qkv_bias);
    }

    #[test]
    fn parse_smollm_config() {
        // SmolLM-135M uses llama architecture
        let json = r#"{
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 576,
            "intermediate_size": 1536,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "vocab_size": 49152,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "torch_dtype": "bfloat16"
        }"#;

        let config = HFConfig::from_json(json.as_bytes()).unwrap();
        let mc = config.to_model_config().unwrap();
        assert_eq!(mc.architecture, Architecture::Llama);
        assert_eq!(mc.hidden_size, 576);
        assert_eq!(mc.head_dim, 64);
    }

    #[test]
    fn detect_architecture_from_architectures_field() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32
        }"#;

        let config = HFConfig::from_json(json.as_bytes()).unwrap();
        assert_eq!(config.detect_architecture(), Some(Architecture::Mistral));
    }

    #[test]
    fn defaults_for_missing_fields() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 8
        }"#;

        let config = HFConfig::from_json(json.as_bytes()).unwrap();
        let mc = config.to_model_config().unwrap();
        // num_kv_heads defaults to num_attention_heads
        assert_eq!(mc.num_kv_heads, 8);
        // intermediate_size defaults to hidden_size * 4
        assert_eq!(mc.intermediate_size, 2048);
        // vocab_size defaults to 32000
        assert_eq!(mc.vocab_size, 32000);
        // max_seq_len defaults to 2048
        assert_eq!(mc.max_seq_len, 2048);
        // dtype defaults to F16
        assert_eq!(mc.dtype, DType::F16);
    }

    #[test]
    fn unknown_architecture_returns_none() {
        let json = r#"{"model_type": "unknown_arch"}"#;
        let config = HFConfig::from_json(json.as_bytes()).unwrap();
        assert_eq!(config.detect_architecture(), None);
    }
}
