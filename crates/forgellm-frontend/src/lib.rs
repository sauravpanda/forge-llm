//! ForgeLLM Frontend — Model parsing and IR construction.
//!
//! This crate handles parsing model formats (GGUF, SafeTensors) and
//! constructing the intermediate representation (IR) used by the
//! optimizer and code generation backends.

pub mod config;
pub mod gguf;
pub mod graph_builder;
pub mod hub;
pub mod ir;
pub mod lora;
pub mod safetensors;
pub mod safetensors_loader;
pub mod weight_loader;

/// Re-export core IR types at the crate root.
pub use ir::*;
pub use lora::{load_lora, load_lora_from_bytes, merge_lora, LoraAdapter, LoraError, LoraLayer};
pub use safetensors_loader::load_safetensors;
