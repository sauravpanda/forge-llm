# Code Generation

ForgeLLM generates standalone Rust source code from IR computation graphs. The generated code has zero dynamic dispatch — every dimension is a compile-time constant.

## CPU Backend

The CPU codegen (`forgellm-codegen-cpu`) emits a complete Rust module containing:

### Constants
All model dimensions are emitted as `const`:
```rust
pub const HIDDEN_SIZE: usize = 576;
pub const NUM_LAYERS: usize = 30;
pub const NUM_HEADS: usize = 9;
pub const VOCAB_SIZE: usize = 49152;
```

### Kernel Functions
Each operation gets a concrete implementation:
- `rms_norm()` — RMS layer normalization
- `matmul()` — Matrix multiplication (M×K × K×N)
- `silu()` — SiLU activation
- `softmax()` — Numerically stable softmax
- `rope()` — Rotary position embedding
- `attention()` — Grouped-query attention with KV cache
- `embedding()` — Token embedding lookup

### Data Structures
```rust
pub struct Weights { ... }       // All model weights
pub struct LayerWeights { ... }  // Per-layer weights
pub struct KVCache { ... }       // KV cache for generation
```

### Forward Function
A `forward()` function chains all operations for single-token generation.

## Future Backends

- **WASM**: SIMD128 + WebGPU companion shaders
- **GPU**: WGSL compute shaders via wgpu
- **CUDA**: Optional PTX emission for maximum performance
