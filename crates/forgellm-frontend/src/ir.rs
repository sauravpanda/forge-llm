//! Intermediate representation for transformer computation graphs.
//!
//! The IR is the central abstraction in Forge: all frontends produce it,
//! all backends consume it.

/// Placeholder for the computation graph IR.
/// Will be fully defined in PR 2.
#[derive(Debug, Clone)]
pub struct Graph {
    pub name: String,
}

impl Graph {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_empty_graph() {
        let graph = Graph::new("test");
        assert_eq!(graph.name, "test");
    }
}
