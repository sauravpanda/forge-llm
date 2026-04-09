//! Tokenizer wrapper — encode text to token IDs and decode back.
//!
//! Uses the HuggingFace `tokenizers` crate, loading from either
//! a `tokenizer.json` file (HF format) or extracting from GGUF metadata.

use std::path::Path;

/// Wrapper around the HuggingFace tokenizer.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

/// Errors during tokenizer operations.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer: {0}")]
    Load(String),

    #[error("encoding failed: {0}")]
    Encode(String),

    #[error("decoding failed: {0}")]
    Decode(String),
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, TokenizerError> {
        let inner: tokenizers::Tokenizer =
            json.parse()
                .map_err(|e: Box<dyn std::error::Error + Send + Sync>| {
                    TokenizerError::Load(e.to_string())
                })?;
        Ok(Self { inner })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text with special tokens (e.g., BOS).
    pub fn encode_with_special(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, true)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    /// Decode a single token ID to text.
    pub fn decode_one(&self, id: u32) -> Result<String, TokenizerError> {
        self.decode(&[id])
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get the token ID for a special token by content (e.g., "<|endoftext|>").
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Get the BOS (beginning of sequence) token ID, if defined.
    pub fn bos_token_id(&self) -> Option<u32> {
        // Common BOS tokens across models
        self.token_to_id("<s>")
            .or_else(|| self.token_to_id("<|begin_of_text|>"))
            .or_else(|| self.token_to_id("<|startoftext|>"))
    }

    /// Get the EOS (end of sequence) token ID, if defined.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.token_to_id("</s>")
            .or_else(|| self.token_to_id("<|end_of_text|>"))
            .or_else(|| self.token_to_id("<|endoftext|>"))
            .or_else(|| self.token_to_id("<|im_end|>"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal tokenizer JSON for testing.
    /// This creates a character-level tokenizer.
    fn minimal_tokenizer_json() -> String {
        r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 2, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": "<unk>",
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": {
                    "<s>": 0, "</s>": 1, "<unk>": 2,
                    "h": 3, "e": 4, "l": 5, "o": 6,
                    "he": 7, "ll": 8, "lo": 9,
                    "hel": 10, "llo": 11
                },
                "merges": [
                    "h e", "l l", "l o", "he l", "ll o"
                ]
            }
        }"#
        .to_string()
    }

    #[test]
    fn load_from_json() {
        let json = minimal_tokenizer_json();
        let tok = Tokenizer::from_json(&json).unwrap();
        assert!(tok.vocab_size() > 0);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let json = minimal_tokenizer_json();
        let tok = Tokenizer::from_json(&json).unwrap();

        let ids = tok.encode("hello").unwrap();
        assert!(!ids.is_empty());

        let text = tok.decode(&ids).unwrap();
        assert_eq!(text, "hello");
    }

    #[test]
    fn special_tokens() {
        let json = minimal_tokenizer_json();
        let tok = Tokenizer::from_json(&json).unwrap();

        assert_eq!(tok.bos_token_id(), Some(0));
        assert_eq!(tok.eos_token_id(), Some(1));
    }

    #[test]
    fn decode_single_token() {
        let json = minimal_tokenizer_json();
        let tok = Tokenizer::from_json(&json).unwrap();

        // Token 3 = "h"
        let text = tok.decode_one(3).unwrap();
        assert!(!text.is_empty());
    }
}
