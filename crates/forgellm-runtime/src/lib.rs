//! Forge Runtime — minimal inference runtime.
//!
//! Provides KV cache management, token sampling,
//! tokenizer integration, and an OpenAI-compatible API server.

/// Placeholder for the runtime.
/// Will be implemented in PR 7.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(!version().is_empty());
    }
}
