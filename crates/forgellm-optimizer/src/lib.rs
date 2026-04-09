//! ForgeLLM Optimizer — graph-level optimization passes.
//!
//! Transforms IR computation graphs to improve performance:
//! - Operator fusion (merge multiple ops into single kernels)
//! - Dead node elimination
//!
//! Each pass is independent and composable.

mod fusion;
mod passes;

pub use passes::optimize;
