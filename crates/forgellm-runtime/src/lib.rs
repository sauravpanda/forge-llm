//! ForgeLLM Runtime — minimal inference runtime.
//!
//! Provides KV cache management, token sampling,
//! and tokenizer integration for compiled models.

pub mod chat;
pub mod interpreter;
pub mod kernels;
pub mod kv_cache;
pub mod sampling;
pub mod tokenizer;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
