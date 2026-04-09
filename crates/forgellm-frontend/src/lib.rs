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
pub mod safetensors;
pub mod weight_loader;

/// Re-export core IR types at the crate root.
pub use ir::*;
