//! Forge Frontend — Model parsing and IR construction.
//!
//! This crate handles parsing model formats (GGUF, SafeTensors) and
//! constructing the intermediate representation (IR) used by the
//! optimizer and code generation backends.

pub mod ir;

/// Re-export core IR types at the crate root.
pub use ir::*;
