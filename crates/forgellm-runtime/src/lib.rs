//! ForgeLLM Runtime — minimal inference runtime.
//!
//! Provides KV cache management, token sampling,
//! and tokenizer integration for compiled models.

pub mod interpreter;
pub mod kv_cache;
pub mod sampling;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
