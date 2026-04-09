# Runtime

The runtime provides components needed to actually run inference: a reference interpreter, KV cache, sampling, and tokenization.

## Reference Interpreter

The interpreter executes IR graphs directly on the CPU using f32 arithmetic. It implements every IR operation as a straightforward Rust function — no SIMD, no fusion, just correct behavior.

This serves two purposes:
1. **Validation**: verify that the model produces correct output
2. **Baseline**: measure performance before optimizations

### Usage

```rust
use forgellm_runtime::{interpreter, kv_cache::KVCache};

let logits = interpreter::forward(token_id, position, &graph, &weights, &mut cache);
```

## KV Cache

The KV cache stores key and value projections for each layer across sequence positions, enabling autoregressive generation without recomputing past tokens.

```rust
let mut cache = KVCache::with_capacity(num_layers, num_kv_heads, head_dim, max_seq_len);

// During generation:
cache.append(layer_idx, &k_data, &v_data);
cache.advance(); // after processing all layers for one token
```

## Sampling

Multiple sampling strategies:

| Strategy | Config |
|----------|--------|
| Greedy | `temperature = 0.0` |
| Temperature | `temperature = 0.7` |
| Top-k | `top_k = 40` |
| Top-p (nucleus) | `top_p = 0.9` |
| Repetition penalty | `repetition_penalty = 1.1` |

## Tokenizer

Wraps the HuggingFace `tokenizers` crate for BPE tokenization:

```rust
let tokenizer = Tokenizer::from_file("tokenizer.json")?;
let ids = tokenizer.encode("Hello world")?;
let text = tokenizer.decode(&ids)?;
```
