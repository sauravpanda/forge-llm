//! ForgeLLM CPU Code Generation — emits Rust source from IR graphs.
//!
//! Takes a computation graph and generates standalone Rust source code
//! with all operations implemented as concrete functions. The generated
//! code has no dynamic dispatch — every operation is specialized to the
//! exact tensor dimensions from the model.

mod emit;
pub mod project;

pub use emit::{generate, CodegenError};
pub use project::{generate_project, ProjectError};
