//! KV cache for autoregressive transformer generation.
//!
//! Stores key and value projections for each layer across sequence positions.
//! Designed for single-sequence generation (batch=1).

/// KV cache for a single model.
///
/// Pre-allocates buffers for the maximum sequence length to avoid
/// allocations during generation.
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Key cache: `[num_layers][max_seq_len * num_kv_heads * head_dim]`
    k: Vec<Vec<f32>>,
    /// Value cache: `[num_layers][max_seq_len * num_kv_heads * head_dim]`
    v: Vec<Vec<f32>>,
    /// Number of layers.
    num_layers: usize,
    /// Number of KV heads.
    num_kv_heads: usize,
    /// Head dimension.
    head_dim: usize,
    /// Current sequence length (number of tokens cached).
    len: usize,
}

impl KVCache {
    /// Create a new empty KV cache.
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            k: (0..num_layers).map(|_| Vec::new()).collect(),
            v: (0..num_layers).map(|_| Vec::new()).collect(),
            num_layers,
            num_kv_heads,
            head_dim,
            len: 0,
        }
    }

    /// Create a new KV cache with pre-allocated capacity.
    pub fn with_capacity(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let entry_size = num_kv_heads * head_dim;
        let capacity = max_seq_len * entry_size;
        Self {
            k: (0..num_layers)
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
            v: (0..num_layers)
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
            num_layers,
            num_kv_heads,
            head_dim,
            len: 0,
        }
    }

    /// Append K and V vectors for the current token to a specific layer.
    ///
    /// `k_data` and `v_data` should each have length `num_kv_heads * head_dim`.
    pub fn append(&mut self, layer: usize, k_data: &[f32], v_data: &[f32]) {
        debug_assert_eq!(k_data.len(), self.num_kv_heads * self.head_dim);
        debug_assert_eq!(v_data.len(), self.num_kv_heads * self.head_dim);
        self.k[layer].extend_from_slice(k_data);
        self.v[layer].extend_from_slice(v_data);
    }

    /// Advance the sequence position by one token.
    /// Call this after appending K/V data to all layers.
    pub fn advance(&mut self) {
        self.len += 1;
    }

    /// Get the full K cache for a layer: `[len * num_kv_heads * head_dim]`.
    pub fn k(&self, layer: usize) -> &[f32] {
        &self.k[layer]
    }

    /// Get the full V cache for a layer: `[len * num_kv_heads * head_dim]`.
    pub fn v(&self, layer: usize) -> &[f32] {
        &self.v[layer]
    }

    /// Current sequence length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Clear the cache for a new generation.
    pub fn clear(&mut self) {
        for layer_k in &mut self.k {
            layer_k.clear();
        }
        for layer_v in &mut self.v {
            layer_v.clear();
        }
        self.len = 0;
    }

    /// Entry size (num_kv_heads * head_dim) per token per layer.
    pub fn entry_size(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_append() {
        let mut cache = KVCache::new(2, 4, 16);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.entry_size(), 64);

        // Append one token to both layers
        let kv_data = vec![1.0f32; 64];
        cache.append(0, &kv_data, &kv_data);
        cache.append(1, &kv_data, &kv_data);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.k(0).len(), 64);
        assert_eq!(cache.v(1).len(), 64);
    }

    #[test]
    fn append_multiple_tokens() {
        let mut cache = KVCache::new(1, 2, 8);

        for token_idx in 0..5 {
            let data = vec![token_idx as f32; 16];
            cache.append(0, &data, &data);
            cache.advance();
        }

        assert_eq!(cache.len(), 5);
        assert_eq!(cache.k(0).len(), 80); // 5 tokens * 2 heads * 8 dim
                                          // First entry should be 0.0, last should be 4.0
        assert_eq!(cache.k(0)[0], 0.0);
        assert_eq!(cache.k(0)[64], 4.0);
    }

    #[test]
    fn clear_cache() {
        let mut cache = KVCache::new(2, 4, 16);
        let data = vec![1.0f32; 64];
        cache.append(0, &data, &data);
        cache.append(1, &data, &data);
        cache.advance();
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.k(0).is_empty());
    }

    #[test]
    fn with_capacity() {
        let cache = KVCache::with_capacity(16, 8, 64, 2048);
        assert!(cache.is_empty());
        assert_eq!(cache.num_layers(), 16);
        assert_eq!(cache.entry_size(), 512);
    }

    // ── Real-world validation tests ──────────────────────────────────────

    #[test]
    fn clear_resets_completely_for_independent_generation() {
        // After clear(), the cache should behave identically to a fresh cache.
        // This matters for multi-turn serving where the model resets between requests.
        let mut cache = KVCache::new(2, 4, 16);

        // Fill cache with some data
        let data_a = vec![1.0f32; 64];
        for _ in 0..5 {
            cache.append(0, &data_a, &data_a);
            cache.append(1, &data_a, &data_a);
            cache.advance();
        }
        assert_eq!(cache.len(), 5);
        assert_eq!(cache.k(0).len(), 5 * 64);

        // Clear and verify complete reset
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.k(0).is_empty());
        assert!(cache.v(0).is_empty());
        assert!(cache.k(1).is_empty());
        assert!(cache.v(1).is_empty());

        // Append new data after clear — should be independent of previous content
        let data_b = vec![2.0f32; 64];
        cache.append(0, &data_b, &data_b);
        cache.append(1, &data_b, &data_b);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.k(0).len(), 64);
        // First element should be from data_b, not data_a
        assert_eq!(
            cache.k(0)[0],
            2.0,
            "after clear, new data should overwrite old content"
        );
    }

    #[test]
    fn cache_handles_max_realistic_sequence() {
        // Simulate filling a cache up to a realistic max sequence length (512 tokens)
        // for a small model and verify no data corruption at the boundary.
        let num_layers = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let max_seq = 512;
        let entry_size = num_kv_heads * head_dim; // 16

        let mut cache = KVCache::with_capacity(num_layers, num_kv_heads, head_dim, max_seq);

        for pos in 0..max_seq {
            let data: Vec<f32> = (0..entry_size)
                .map(|j| (pos * entry_size + j) as f32)
                .collect();
            for layer in 0..num_layers {
                cache.append(layer, &data, &data);
            }
            cache.advance();
        }

        assert_eq!(cache.len(), max_seq);
        assert_eq!(cache.k(0).len(), max_seq * entry_size);

        // Verify data integrity at first and last positions
        // First token in layer 0: values 0..16
        assert_eq!(cache.k(0)[0], 0.0);
        assert_eq!(cache.k(0)[entry_size - 1], (entry_size - 1) as f32);
        // Last token in layer 0: values (511*16)..(511*16+16)
        let last_start = (max_seq - 1) * entry_size;
        assert_eq!(cache.k(0)[last_start], last_start as f32);
    }
}
